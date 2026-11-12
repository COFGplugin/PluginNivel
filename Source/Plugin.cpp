// NivelPlugin - VST3 mono, nivelacion de volumen por secciones (clip gain).
#include <juce_audio_utils/juce_audio_utils.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace
{
    // Capacidad fija del buffer de captura (muestras). 300 s a 48 kHz.
    constexpr int64_t kMaxCaptureSamples = 48000LL * 300LL;
    constexpr int     kMaxSections       = 1024;
    constexpr int     kMaxUndoSteps      = 100;
    constexpr float   kSilenceFloor      = 1.0e-4f;   // -80 dBFS: por debajo no se toca
    constexpr int64_t kNoPlayhead        = std::numeric_limits<int64_t>::min();
    constexpr int     kRulerH            = 18;
    constexpr int     kScrollH           = 14;

    juce::String T (const char* utf8) { return juce::String::fromUTF8 (utf8); }

    // m:ss.mmm (decimals = 0..3)
    juce::String formatTime (double sec, int decimals)
    {
        if (sec < 0.0) sec = 0.0;
        const int m = (int) std::floor (sec / 60.0);
        const double s = sec - (double) m * 60.0;
        juce::String ss = juce::String (s, decimals);
        if (s < 10.0) ss = "0" + ss;
        return juce::String (m) + ":" + ss;
    }

    //==========================================================================
    struct GainMap
    {
        int     count  = 0;
        int64_t length = 0;
        int64_t minLen = 0;
        int64_t start[kMaxSections] {};
        float   gain[kMaxSections]  {};
    };

    // Ganancia lineal en la muestra t (relativa al inicio de la captura).
    // Constante dentro de cada seccion; rampa lineal de 'fade' muestras
    // centrada en cada corte.
    inline float gainAt (const GainMap& m, int64_t t, int64_t fade) noexcept
    {
        if (m.count <= 0 || t < 0 || t >= m.length)
            return 1.0f;

        int lo = 0, hi = m.count - 1;
        while (lo < hi)
        {
            const int mid = (lo + hi + 1) / 2;
            if (m.start[mid] <= t) lo = mid; else hi = mid - 1;
        }

        const float g = m.gain[lo];

        if (fade > 1)
        {
            const int64_t half = fade / 2;

            if (lo > 0)
            {
                const int64_t d = t - (m.start[lo] - half);
                if (d >= 0 && d < fade)
                {
                    const float f  = ((float) d + 0.5f) / (float) fade;
                    const float gp = m.gain[lo - 1];
                    return gp + (g - gp) * f;
                }
            }

            if (lo + 1 < m.count)
            {
                const int64_t d = t - (m.start[lo + 1] - half);
                if (d >= 0 && d < fade)
                {
                    const float f  = ((float) d + 0.5f) / (float) fade;
                    const float gn = m.gain[lo + 1];
                    return g + (gn - g) * f;
                }
            }
        }
        return g;
    }
}

//==============================================================================
class NivelPluginEditor;

class NivelPluginProcessor : public juce::AudioProcessor
{
public:
    NivelPluginProcessor()
        : AudioProcessor (BusesProperties()
                              .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                              .withOutput ("Output", juce::AudioChannelSet::mono(), true)),
          apvts (*this, nullptr, "STATE", createLayout())
    {
        capture.assign ((size_t) kMaxCaptureSamples, 0.0f);

        targetParam    = apvts.getRawParameterValue ("target");
        fadeParam      = apvts.getRawParameterValue ("fade");
        zeroCrossParam = apvts.getRawParameterValue ("zerocross");
        threshParam    = apvts.getRawParameterValue ("silthresh");
        minSilParam    = apvts.getRawParameterValue ("minsil");
        applyParam     = apvts.getRawParameterValue ("apply");
        autoStopParam  = apvts.getRawParameterValue ("autostop");
    }

    //==========================================================================
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout layout;

        layout.add (std::make_unique<AudioParameterFloat> (
            ParameterID { "target", 1 }, "Nivel Objetivo",
            NormalisableRange<float> (-60.0f, 0.0f, 0.1f), -18.0f,
            AudioParameterFloatAttributes().withLabel ("dBFS")));

        layout.add (std::make_unique<AudioParameterFloat> (
            ParameterID { "fade", 1 }, "Crossfade",
            NormalisableRange<float> (0.0f, 5.0f, 0.1f), 5.0f,
            AudioParameterFloatAttributes().withLabel ("ms")));

        layout.add (std::make_unique<AudioParameterBool> (
            ParameterID { "zerocross", 1 }, "Cortes en cruce por cero", true));

        layout.add (std::make_unique<AudioParameterFloat> (
            ParameterID { "silthresh", 1 }, "Umbral de silencio",
            NormalisableRange<float> (-90.0f, -10.0f, 1.0f), -40.0f,
            AudioParameterFloatAttributes().withLabel ("dBFS")));

        layout.add (std::make_unique<AudioParameterFloat> (
            ParameterID { "minsil", 1 }, "Silencio minimo",
            NormalisableRange<float> (20.0f, 1000.0f, 10.0f), 150.0f,
            AudioParameterFloatAttributes().withLabel ("ms")));

        layout.add (std::make_unique<AudioParameterBool> (
            ParameterID { "apply", 1 }, "Aplicar ganancia", true));

        layout.add (std::make_unique<AudioParameterBool> (
            ParameterID { "autostop", 1 }, "Detener captura al parar", true));

        return layout;
    }

    //==========================================================================
    void prepareToPlay (double sampleRate, int) override
    {
        currentSR.store (sampleRate);
        const double csr = captureSR.load();
        if (csr > 0.0 && std::abs (csr - sampleRate) > 0.5)
            clearAll();
    }

    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& l) const override
    {
        return l.getMainInputChannelSet()  == juce::AudioChannelSet::mono()
            && l.getMainOutputChannelSet() == juce::AudioChannelSet::mono();
    }

    //==========================================================================
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;

        const int n = buffer.getNumSamples();
        if (buffer.getNumChannels() < 1 || n <= 0)
            return;

        for (int ch = 1; ch < buffer.getNumChannels(); ++ch)
            buffer.clear (ch, 0, n);

        float* data = buffer.getWritePointer (0);

        // --- posicion del host ---
        int64_t hostPos = 0;
        bool havePos = false, playing = false;

        if (auto* ph = getPlayHead())
        {
            const auto infoOpt = ph->getPosition();
            if (infoOpt.hasValue())
            {
                const auto& info = *infoOpt;
                playing = info.getIsPlaying();
                const auto t = info.getTimeInSamples();
                if (t.hasValue())
                {
                    hostPos = *t;
                    havePos = true;
                }
            }
        }

        hostPlaying.store (playing);
        playheadIdx.store ((havePos && hasStart.load()) ? hostPos - captureStartHost.load()
                                                        : kNoPlayhead);

        // --- CAPTURA (el audio pasa sin modificar) ---
        if (capturing.load())
        {
            int64_t idx = 0;
            bool canWrite = false;

            if (havePos)
            {
                if (playing)
                {
                    if (! hasStart.load())
                    {
                        captureStartHost.store (hostPos);
                        hasStart.store (true);
                    }
                    idx = hostPos - captureStartHost.load();
                    canWrite = true;
                }
                else if (hasStart.load() && autoStopParam->load() > 0.5f)
                {
                    // el host paro: termina la captura
                    capturing.store (false);
                    ++revision;
                }
            }
            else
            {
                idx = seqPos.load();
                seqPos.store (idx + n);
                canWrite = true;
            }

            if (canWrite)
            {
                float* cap = capture.data();
                for (int i = 0; i < n; ++i)
                {
                    const int64_t j = idx + i;
                    if (j >= 0 && j < kMaxCaptureSamples)
                        cap[j] = data[i];
                }

                const int64_t end = std::min<int64_t> (idx + n, kMaxCaptureSamples);
                if (end > captureLen.load())
                    captureLen.store (end);

                if (idx + n >= kMaxCaptureSamples)
                    capturing.store (false);
            }
            return;
        }

        // --- REPRODUCCION: solo multiplicacion por ganancia ---
        if (applyParam->load() < 0.5f)
            return;
        if (! havePos || ! playing || ! hasStart.load())
            return;

        const uint32_t v = mapVersion.load (std::memory_order_acquire);
        if (v != cachedVersion)
        {
            const juce::SpinLock::ScopedTryLockType lock (mapLock);
            if (lock.isLocked())
            {
                cachedMap = sharedMap;
                cachedVersion = v;
            }
        }

        if (cachedMap.count <= 0)
            return;

        const int64_t base = hostPos - captureStartHost.load();
        if (base + n <= 0 || base >= cachedMap.length)
            return;

        const int64_t fade = std::min<int64_t> (
            cachedMap.minLen,
            (int64_t) std::llround ((double) fadeParam->load() * 0.001 * currentSR.load()));

        const bool useBuffer = bufferValid.load();
        const float* cap = capture.data();

        for (int i = 0; i < n; ++i)
        {
            const int64_t j = base + i;
            const float g = gainAt (cachedMap, j, fade);

            if (useBuffer && j >= 0 && j < cachedMap.length)
                data[i] = cap[j] * g;     // reproduce lo capturado
            else
                data[i] *= g;             // fuera de la captura / audio no guardado
        }
    }

    //==========================================================================
    // Acciones de la GUI (hilo de mensajes)
    void startCapture()
    {
        capturing.store (false);
        captureLen.store (0);
        hasStart.store (false);
        seqPos.store (0);
        cuts.clear();
        cutsDirty = false;
        undoStack.clear();
        redoStack.clear();
        setGainMap (GainMap(), false);
        std::fill (capture.begin(), capture.end(), 0.0f);
        captureSR.store (currentSR.load());
        bufferValid.store (true);
        ++revision;
        capturing.store (true);
    }

    void stopCapture()
    {
        capturing.store (false);
        ++revision;
    }

    void clearAll()
    {
        capturing.store (false);
        captureLen.store (0);
        hasStart.store (false);
        seqPos.store (0);
        cuts.clear();
        cutsDirty = false;
        undoStack.clear();
        redoStack.clear();
        setGainMap (GainMap(), false);
        std::fill (capture.begin(), capture.end(), 0.0f);
        bufferValid.store (false);
        captureSR.store (0.0);
        ++revision;
    }

    bool analyze()
    {
        if (! bufferValid.load())
            return false;

        stopCapture();
        const int64_t len = captureLen.load();
        if (len <= 0)
            return false;

        pushUndo();

        if (zeroCrossParam->load() > 0.5f)
            for (auto& c : cuts)
                c = snapToZeroCrossing (c);

        normaliseCuts (len);

        std::vector<int64_t> b;
        b.push_back (0);
        b.insert (b.end(), cuts.begin(), cuts.end());
        b.push_back (len);

        GainMap m;
        m.length = len;
        m.count  = (int) b.size() - 1;
        m.minLen = len;

        const double targetLin = juce::Decibels::decibelsToGain ((double) targetParam->load());

        for (int k = 0; k < m.count; ++k)
        {
            const int64_t s = b[(size_t) k], e = b[(size_t) k + 1];

            float peak = 0.0f;                       // pico maximo absoluto
            for (int64_t j = s; j < e; ++j)
                peak = std::max (peak, std::abs (capture[(size_t) j]));

            m.start[k] = s;
            m.gain[k]  = (peak > kSilenceFloor) ? (float) (targetLin / (double) peak) : 1.0f;
            m.minLen   = std::min (m.minLen, e - s);
        }

        setGainMap (m, true);
        cutsDirty = false;
        ++revision;
        return true;
    }

    void autoSplit()
    {
        if (! bufferValid.load())
            return;

        stopCapture();
        const int64_t len = captureLen.load();
        const double  sr  = currentSR.load();
        const int64_t win = std::max<int64_t> (16, (int64_t) (0.020 * sr));
        const int64_t hop = std::max<int64_t> (8, win / 2);
        if (len < win * 4)
            return;

        pushUndo();

        const float thr = (float) juce::Decibels::decibelsToGain ((double) threshParam->load());
        const int minRun = std::max (1, (int) std::llround ((double) minSilParam->load() * 0.001 * sr / (double) hop));

        std::vector<float> env;
        env.reserve ((size_t) (len / hop) + 1);
        for (int64_t p = 0; p + win <= len; p += hop)
        {
            double acc = 0.0;
            for (int64_t j = p; j < p + win; ++j)
            {
                const double x = (double) capture[(size_t) j];
                acc += x * x;
            }
            env.push_back ((float) std::sqrt (acc / (double) win));
        }

        std::vector<int64_t> found;
        int runStart = -1;
        for (size_t i = 0; i <= env.size(); ++i)
        {
            const bool silent = (i < env.size()) && env[i] < thr;
            if (silent)
            {
                if (runStart < 0) runStart = (int) i;
            }
            else if (runStart >= 0)
            {
                const int runEnd = (int) i - 1;
                const int runLen = runEnd - runStart + 1;
                if (runLen >= minRun && runStart > 0 && i < env.size())
                    found.push_back ((int64_t) ((runStart + runEnd) / 2) * hop + win / 2);
                runStart = -1;
            }
        }

        cuts = std::move (found);
        if (zeroCrossParam->load() > 0.5f)
            for (auto& c : cuts)
                c = snapToZeroCrossing (c);

        normaliseCuts (len);
        cutsDirty = true;
        ++revision;
    }

    void addCut (int64_t pos)
    {
        const int64_t len = captureLen.load();
        if (pos <= 0 || pos >= len || (int) cuts.size() >= kMaxSections - 1)
            return;
        pushUndo();
        cuts.push_back (pos);
        std::sort (cuts.begin(), cuts.end());
        cutsDirty = true;
        ++revision;
    }

    void removeCut (int i)
    {
        if (i < 0 || i >= (int) cuts.size())
            return;
        pushUndo();
        cuts.erase (cuts.begin() + i);
        cutsDirty = true;
        ++revision;
    }

    // Llamar una vez al empezar a arrastrar un corte (guarda un paso de deshacer)
    void beginEdit() { pushUndo(); }

    void moveCut (int i, int64_t pos)
    {
        if (i < 0 || i >= (int) cuts.size())
            return;
        const int64_t len = captureLen.load();
        const int64_t lo = (i > 0 ? cuts[(size_t) i - 1] : 0) + 1;
        const int64_t hi = (i + 1 < (int) cuts.size() ? cuts[(size_t) i + 1] : len) - 1;
        if (hi < lo)
            return;
        cuts[(size_t) i] = juce::jlimit (lo, hi, pos);
        cutsDirty = true;
        ++revision;
    }

    void snapCut (int i)
    {
        if (i < 0 || i >= (int) cuts.size() || zeroCrossParam->load() < 0.5f)
            return;
        const int64_t len = captureLen.load();
        const int64_t lo = (i > 0 ? cuts[(size_t) i - 1] : 0) + 1;
        const int64_t hi = (i + 1 < (int) cuts.size() ? cuts[(size_t) i + 1] : len) - 1;
        if (hi < lo)
            return;
        cuts[(size_t) i] = juce::jlimit (lo, hi, snapToZeroCrossing (cuts[(size_t) i]));
        ++revision;
    }

    int64_t snapIfEnabled (int64_t pos) const
    {
        return zeroCrossParam->load() > 0.5f ? snapToZeroCrossing (pos) : pos;
    }

    //==========================================================================
    // Deshacer / rehacer (cortes y analisis; la captura de audio no se deshace)
    bool canUndo() const noexcept { return ! undoStack.empty(); }
    bool canRedo() const noexcept { return ! redoStack.empty(); }

    void undo()
    {
        if (undoStack.empty())
            return;
        redoStack.push_back (makeSnapshot());
        const Snapshot s = undoStack.back();
        undoStack.pop_back();
        restoreSnapshot (s);
    }

    void redo()
    {
        if (redoStack.empty())
            return;
        undoStack.push_back (makeSnapshot());
        const Snapshot s = redoStack.back();
        redoStack.pop_back();
        restoreSnapshot (s);
    }

    //==========================================================================
    // Acceso de solo lectura para la GUI
    const std::vector<float>& getCapture() const noexcept { return capture; }
    int64_t  getCaptureLength() const noexcept { return captureLen.load(); }
    bool     isCapturing()      const noexcept { return capturing.load(); }
    bool     isBufferValid()    const noexcept { return bufferValid.load(); }
    bool     isAnalyzed()       const noexcept { return analyzed.load(); }
    bool     areCutsDirty()     const noexcept { return cutsDirty; }
    bool     isHostPlaying()    const noexcept { return hostPlaying.load(); }
    bool     isHostStartKnown() const noexcept { return hasStart.load(); }
    int64_t  getCaptureStartHost() const noexcept { return captureStartHost.load(); }
    int64_t  getPlayheadIndex() const noexcept { return playheadIdx.load(); }
    uint32_t getRevision()      const noexcept { return revision.load(); }
    double   getSampleRate()    const noexcept { return currentSR.load(); }
    float    getFadeMs()        const noexcept { return fadeParam->load(); }
    std::vector<int64_t> getCuts() const { return cuts; }

    GainMap getGainMapCopy() const
    {
        const juce::SpinLock::ScopedLockType lock (mapLock);
        return sharedMap;
    }

    //==========================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "NivelPlugin"; }
    bool acceptsMidi() const override  { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    //==========================================================================
    void getStateInformation (juce::MemoryBlock& dest) override
    {
        auto state = apvts.copyState();

        if (analyzed.load())
        {
            const auto m = getGainMapCopy();
            juce::String s;
            for (int k = 0; k < m.count; ++k)
                s << juce::String ((juce::int64) m.start[k]) << ":"
                  << juce::String ((double) m.gain[k], 9) << ";";

            state.setProperty ("gmap",   s, nullptr);
            state.setProperty ("glen",   (juce::int64) m.length, nullptr);
            state.setProperty ("cstart", (juce::int64) captureStartHost.load(), nullptr);
            state.setProperty ("csr",    captureSR.load(), nullptr);
        }

        if (auto xml = state.createXml())
            copyXmlToBinary (*xml, dest);
    }

    void setStateInformation (const void* data, int size) override
    {
        std::unique_ptr<juce::XmlElement> xml (getXmlFromBinary (data, size));
        if (xml == nullptr || ! xml->hasTagName (apvts.state.getType()))
            return;

        const auto vt = juce::ValueTree::fromXml (*xml);
        apvts.replaceState (vt);

        const juce::String s = vt.getProperty ("gmap").toString();
        if (s.isEmpty())
            return;

        GainMap m;
        m.length = (int64_t) (juce::int64) vt.getProperty ("glen", 0);

        for (const auto& tok : juce::StringArray::fromTokens (s, ";", ""))
        {
            const auto kv = juce::StringArray::fromTokens (tok, ":", "");
            if (kv.size() == 2 && m.count < kMaxSections)
            {
                m.start[m.count] = (int64_t) kv[0].getLargeIntValue();
                m.gain[m.count]  = (float) kv[1].getDoubleValue();
                ++m.count;
            }
        }

        if (m.count <= 0 || m.length <= 0)
            return;

        m.minLen = m.length;
        for (int k = 0; k < m.count; ++k)
        {
            const int64_t e = (k + 1 < m.count) ? m.start[k + 1] : m.length;
            m.minLen = std::min (m.minLen, e - m.start[k]);
        }

        capturing.store (false);
        captureLen.store (m.length);
        captureStartHost.store ((int64_t) (juce::int64) vt.getProperty ("cstart", 0));
        hasStart.store (true);
        captureSR.store ((double) vt.getProperty ("csr", 0.0));
        bufferValid.store (false);   // el audio original no se guarda en el proyecto

        cuts.clear();
        for (int k = 1; k < m.count; ++k)
            cuts.push_back (m.start[k]);

        undoStack.clear();
        redoStack.clear();
        setGainMap (m, true);
        cutsDirty = false;
        ++revision;
    }

    juce::AudioProcessorValueTreeState apvts;

private:
    struct Snapshot
    {
        std::vector<int64_t> cuts;
        GainMap map;
        bool analyzed = false;
        bool dirty = false;
    };

    Snapshot makeSnapshot() const
    {
        Snapshot s;
        s.cuts     = cuts;
        s.map      = getGainMapCopy();
        s.analyzed = analyzed.load();
        s.dirty    = cutsDirty;
        return s;
    }

    void pushUndo()
    {
        undoStack.push_back (makeSnapshot());
        if ((int) undoStack.size() > kMaxUndoSteps)
            undoStack.erase (undoStack.begin());
        redoStack.clear();
    }

    void restoreSnapshot (const Snapshot& s)
    {
        cuts = s.cuts;
        setGainMap (s.map, s.analyzed);
        cutsDirty = s.dirty;
        ++revision;
    }

    void setGainMap (const GainMap& m, bool isAnalyzed)
    {
        {
            const juce::SpinLock::ScopedLockType lock (mapLock);
            sharedMap = m;
            mapVersion.fetch_add (1, std::memory_order_release);
        }
        analyzed.store (isAnalyzed);
    }

    int64_t snapToZeroCrossing (int64_t idx) const
    {
        const int64_t len = captureLen.load();
        const int64_t win = std::max<int64_t> (1, (int64_t) (0.010 * currentSR.load()));
        const float* c = capture.data();

        auto crosses = [c, len] (int64_t i)
        {
            if (i < 0 || i + 1 >= len)
                return false;
            return (c[i] <= 0.0f && c[i + 1] > 0.0f) || (c[i] >= 0.0f && c[i + 1] < 0.0f);
        };

        for (int64_t d = 0; d <= win; ++d)
        {
            if (crosses (idx + d)) return idx + d + 1;
            if (crosses (idx - d)) return idx - d + 1;
        }
        return idx;
    }

    void normaliseCuts (int64_t len)
    {
        const int64_t minGap = std::max<int64_t> (1, (int64_t) (0.010 * currentSR.load()));
        std::sort (cuts.begin(), cuts.end());

        std::vector<int64_t> out;
        int64_t last = 0;
        for (const auto c : cuts)
        {
            if (c - last >= minGap && len - c >= minGap && (int) out.size() < kMaxSections - 1)
            {
                out.push_back (c);
                last = c;
            }
        }
        cuts = std::move (out);
    }

    //==========================================================================
    std::vector<float> capture;
    std::vector<int64_t> cuts;      // cortes interiores (solo hilo de mensajes)
    bool cutsDirty = false;

    std::vector<Snapshot> undoStack, redoStack;   // solo hilo de mensajes

    std::atomic<bool>     capturing { false };
    std::atomic<bool>     bufferValid { false };
    std::atomic<bool>     analyzed { false };
    std::atomic<bool>     hasStart { false };
    std::atomic<bool>     hostPlaying { false };
    std::atomic<int64_t>  captureLen { 0 };
    std::atomic<int64_t>  captureStartHost { 0 };
    std::atomic<int64_t>  seqPos { 0 };
    std::atomic<int64_t>  playheadIdx { kNoPlayhead };
    std::atomic<double>   currentSR { 48000.0 };
    std::atomic<double>   captureSR { 0.0 };
    std::atomic<uint32_t> revision { 0 };
    std::atomic<uint32_t> mapVersion { 0 };

    mutable juce::SpinLock mapLock;
    GainMap sharedMap;
    GainMap cachedMap;              // copia usada solo por el hilo de audio
    uint32_t cachedVersion = 0;

    std::atomic<float>* targetParam    = nullptr;
    std::atomic<float>* fadeParam      = nullptr;
    std::atomic<float>* zeroCrossParam = nullptr;
    std::atomic<float>* threshParam    = nullptr;
    std::atomic<float>* minSilParam    = nullptr;
    std::atomic<float>* applyParam     = nullptr;
    std::atomic<float>* autoStopParam  = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NivelPluginProcessor)
};

//==============================================================================
class WaveformView : public juce::Component,
                     private juce::Timer,
                     private juce::ScrollBar::Listener
{
public:
    explicit WaveformView (NivelPluginProcessor& p) : proc (p)
    {
        viewSpan = proc.getSampleRate() * 2.0;

        addAndMakeVisible (scrollBar);
        scrollBar.setAutoHide (false);
        scrollBar.addListener (this);

        startTimerHz (30);
    }

    ~WaveformView() override
    {
        stopTimer();
        scrollBar.removeListener (this);
    }

    // Zoom horizontal (eje X)
    void zoomIn()  { zoomBy (0.5, viewStart + viewSpan * 0.5); }
    void zoomOut() { zoomBy (2.0, viewStart + viewSpan * 0.5); }

    void fitView()
    {
        autoFit   = true;
        viewStart = 0.0;
        viewSpan  = std::max ((double) proc.getCaptureLength(), proc.getSampleRate() * 2.0);
        cacheDirty = true;
        updateScrollBar();
    }

    void setFollow (bool f) { follow = f; }

    //==========================================================================
    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff15181c));

        auto bounds = getLocalBounds();
        const auto ruler = bounds.removeFromTop (kRulerH);
        bounds.removeFromBottom (kScrollH);
        const auto area = bounds;

        drawRuler (g, ruler);

        g.setColour (juce::Colour (0xff1b1f25));
        g.fillRect (area);
        g.setColour (juce::Colour (0xff2c323b));
        g.drawHorizontalLine (area.getCentreY(), 0.0f, (float) area.getWidth());

        const int64_t len = proc.getCaptureLength();
        const bool valid  = proc.isBufferValid();

        if (len > 0 && valid && area.getWidth() > 0 && area.getHeight() > 0)
        {
            if (cacheDirty || cache.getWidth() != area.getWidth() || cache.getHeight() != area.getHeight())
                rebuildCache (area.getWidth(), area.getHeight());
            g.drawImageAt (cache, area.getX(), area.getY());
        }
        else
        {
            g.setColour (juce::Colour (0xff7d8794));
            g.setFont (15.0f);
            if (len <= 0)
                g.drawText (T ("Pulsa «Capturar Audio» y reproduce la pista en el DAW"),
                            area, juce::Justification::centred);
            else
                g.drawText (T ("Audio original no guardado en el proyecto: se conservan cortes y ganancias"),
                            area, juce::Justification::centred);
        }

        if (len > 0)
        {
            const auto cutsCopy = proc.getCuts();
            const bool showGains = proc.isAnalyzed() && ! proc.areCutsDirty();
            const auto map = proc.getGainMapCopy();

            std::vector<int64_t> b;
            b.push_back (0);
            b.insert (b.end(), cutsCopy.begin(), cutsCopy.end());
            b.push_back (len);

            // etiquetas de ganancia por seccion
            if (showGains && map.count == (int) b.size() - 1)
            {
                g.setFont (12.0f);
                for (int k = 0; k < map.count; ++k)
                {
                    const float xs = std::max (0.0f, sampleToX ((double) b[(size_t) k]));
                    const float xe = std::min ((float) getWidth(), sampleToX ((double) b[(size_t) k + 1]));
                    if (xe - xs < 56.0f)
                        continue;
                    const float db = juce::Decibels::gainToDecibels (map.gain[k]);
                    g.setColour (juce::Colour (0xffffd166));
                    g.drawText ((db >= 0 ? "+" : "") + juce::String (db, 1) + " dB",
                                juce::Rectangle<float> (xs + 6.0f, (float) area.getY() + 2.0f, xe - xs - 8.0f, 14.0f),
                                juce::Justification::centredLeft, false);
                }
            }

            // fin de la captura
            const float xEnd = sampleToX ((double) len);
            if (xEnd >= 0.0f && xEnd <= (float) getWidth())
            {
                g.setColour (juce::Colour (0xff4a5360));
                g.drawVerticalLine ((int) xEnd, (float) area.getY(), (float) area.getBottom());
            }

            // cortes
            for (const auto c : cutsCopy)
            {
                const float x = sampleToX ((double) c);
                if (x < -6.0f || x > (float) getWidth() + 6.0f)
                    continue;
                g.setColour (juce::Colour (0xffff8c42));
                g.drawVerticalLine ((int) x, (float) area.getY(), (float) area.getBottom());
                juce::Path tri;
                tri.addTriangle (x - 5.0f, (float) area.getY(), x + 5.0f, (float) area.getY(),
                                 x, (float) area.getY() + 8.0f);
                g.fillPath (tri);
            }

            // cursor (se pone haciendo clic en la regla de tiempo)
            if (cursorIdx >= 0)
            {
                const float x = sampleToX ((double) cursorIdx);
                if (x >= 0.0f && x <= (float) getWidth())
                {
                    g.setColour (juce::Colour (0xff7ee787));
                    g.drawVerticalLine ((int) x, (float) ruler.getY(), (float) area.getBottom());
                    juce::Path tri;
                    tri.addTriangle (x - 5.0f, (float) ruler.getY(), x + 5.0f, (float) ruler.getY(),
                                     x, (float) ruler.getBottom());
                    g.fillPath (tri);
                }
            }
        }

        // playhead del host
        const int64_t ph = proc.getPlayheadIndex();
        if (ph != kNoPlayhead)
        {
            const float x = sampleToX ((double) ph);
            if (x >= 0.0f && x <= (float) getWidth())
            {
                g.setColour (juce::Colours::white.withAlpha (0.9f));
                g.fillRect (x - 0.5f, (float) area.getY(), 1.5f, (float) area.getHeight());
            }
        }

        // estado
        juce::String status;
        const double sr = std::max (1.0, proc.getSampleRate());
        if (proc.isCapturing())
            status = T ("● Capturando… ") + juce::String ((double) len / sr, 1) + " s";
        else if (len > 0)
        {
            status = juce::String ((double) len / sr, 1) + " s  |  " + juce::String (proc.getCuts().size() + 1)
                   + T (" secciones  |  ");
            status += proc.areCutsDirty() ? T ("cortes sin analizar")
                                          : (proc.isAnalyzed() ? T ("analizado") : T ("sin analizar"));

            if (cursorIdx >= 0)
            {
                const double off = proc.isHostStartKnown() ? (double) proc.getCaptureStartHost() / sr : 0.0;
                status += "  |  Cursor: " + formatTime (off + (double) cursorIdx / sr, 3);
            }
        }

        auto tmp = area;
        const auto statusR = tmp.removeFromBottom (18).reduced (6, 0);
        g.setColour (proc.isCapturing() ? juce::Colour (0xffff5c5c) : juce::Colour (0xff9aa4b1));
        g.setFont (12.0f);
        g.drawText (status, statusR, juce::Justification::centredRight, false);
    }

    void resized() override
    {
        auto b = getLocalBounds();
        scrollBar.setBounds (b.removeFromBottom (kScrollH));
        cacheDirty = true;
        updateScrollBar();
    }

    //==========================================================================
    void mouseDown (const juce::MouseEvent& e) override
    {
        lastDragX = e.x;
        dragCut = -1;
        dragPushed = false;
        draggingCursor = false;

        if (proc.getCaptureLength() <= 0 || ! proc.isBufferValid())
            return;

        if (e.y < kRulerH && ! e.mods.isPopupMenu())
        {
            draggingCursor = true;
            setCursorFromX ((float) e.x);
            return;
        }

        const int k = findCutNear ((float) e.x);
        if (e.mods.isPopupMenu())
        {
            if (k >= 0)
                proc.removeCut (k);
            return;
        }
        dragCut = k;
    }

    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (e.mods.isPopupMenu())
            return;

        if (draggingCursor)
        {
            setCursorFromX ((float) e.x);
            return;
        }

        if (dragCut >= 0)
        {
            if (! dragPushed)
            {
                proc.beginEdit();
                dragPushed = true;
            }
            proc.moveCut (dragCut, (int64_t) xToSample ((float) e.x));
        }
        else
        {
            viewStart -= (double) (e.x - lastDragX) * viewSpan / (double) std::max (1, getWidth());
            lastDragX = e.x;
            autoFit = false;
            clampView();
        }
    }

    void mouseUp (const juce::MouseEvent&) override
    {
        if (dragCut >= 0 && dragPushed)
            proc.snapCut (dragCut);
        dragCut = -1;
        dragPushed = false;
        draggingCursor = false;
    }

    void mouseDoubleClick (const juce::MouseEvent& e) override
    {
        if (e.y < kRulerH)
            return;
        if (proc.getCaptureLength() <= 0 || ! proc.isBufferValid())
            return;
        proc.addCut (proc.snapIfEnabled ((int64_t) xToSample ((float) e.x)));
    }

    void mouseWheelMove (const juce::MouseEvent& e, const juce::MouseWheelDetails& w) override
    {
        if (e.mods.isShiftDown())
        {
            viewStart -= (double) w.deltaY * viewSpan * 0.25;
            autoFit = false;
            clampView();
        }
        else
        {
            zoomBy (w.deltaY > 0.0f ? 0.8 : 1.25, (double) xToSample ((float) e.x));
        }
    }

private:
    void scrollBarMoved (juce::ScrollBar*, double newRangeStart) override
    {
        viewStart = newRangeStart;
        autoFit = false;
        cacheDirty = true;
    }

    void updateScrollBar()
    {
        scrollBar.setRangeLimits (0.0, maxSpan());
        scrollBar.setCurrentRange (viewStart, viewSpan, juce::dontSendNotification);
    }

    void setCursorFromX (float x)
    {
        const int64_t len = proc.getCaptureLength();
        cursorIdx = juce::jlimit ((int64_t) 0, len, (int64_t) xToSample (x));
    }

    double xToSample (float x) const
    {
        return viewStart + (double) x * viewSpan / (double) std::max (1, getWidth());
    }

    float sampleToX (double s) const
    {
        return (float) ((s - viewStart) * (double) getWidth() / viewSpan);
    }

    double maxSpan() const
    {
        return std::max ((double) proc.getCaptureLength(), proc.getSampleRate() * 2.0);
    }

    void clampView()
    {
        viewSpan  = juce::jlimit (32.0, maxSpan(), viewSpan);
        viewStart = juce::jlimit (0.0, std::max (0.0, (double) proc.getCaptureLength() - viewSpan), viewStart);
        cacheDirty = true;
        updateScrollBar();
    }

    void zoomBy (double factor, double anchor)
    {
        const double newSpan = juce::jlimit (32.0, maxSpan(), viewSpan * factor);
        const double frac = (anchor - viewStart) / viewSpan;
        viewStart = anchor - frac * newSpan;
        viewSpan  = newSpan;
        autoFit = false;
        clampView();
    }

    int findCutNear (float x) const
    {
        const auto c = proc.getCuts();
        int best = -1;
        float bestD = 7.0f;
        for (int i = 0; i < (int) c.size(); ++i)
        {
            const float d = std::abs (sampleToX ((double) c[(size_t) i]) - x);
            if (d < bestD) { bestD = d; best = i; }
        }
        return best;
    }

    void timerCallback() override
    {
        const auto rev = proc.getRevision();
        const auto len = proc.getCaptureLength();
        const auto fm  = proc.getFadeMs();

        if (len <= 0)
            cursorIdx = -1;

        if (autoFit && len != lastLen)
            fitView();

        // seguir el playhead del host cuando hay zoom
        if (follow && ! autoFit && proc.isHostPlaying())
        {
            const int64_t ph = proc.getPlayheadIndex();
            if (ph != kNoPlayhead && ph >= 0 && ph <= len
                && ((double) ph < viewStart || (double) ph > viewStart + viewSpan))
            {
                viewStart = (double) ph - viewSpan * 0.1;
                clampView();
            }
        }

        if (rev != lastRev || len != lastLen || fm != lastFade)
        {
            cacheDirty = true;
            lastRev = rev;
            lastLen = len;
            lastFade = fm;
            updateScrollBar();
        }
        repaint();
    }

    void drawRuler (juce::Graphics& g, juce::Rectangle<int> r) const
    {
        g.setColour (juce::Colour (0xff22262c));
        g.fillRect (r);

        const double sr = proc.getSampleRate();
        if (sr <= 0.0 || getWidth() <= 0)
            return;

        const double secPerPx = viewSpan / sr / (double) getWidth();
        static const double steps[] = { 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300 };
        const int nSteps = (int) (sizeof (steps) / sizeof (steps[0]));

        double step = steps[nSteps - 1];
        for (int i = 0; i < nSteps; ++i)
            if (steps[i] / secPerPx >= 70.0) { step = steps[i]; break; }

        // la regla muestra el tiempo del host (igual que el contador del DAW)
        const double off = proc.isHostStartKnown() ? (double) proc.getCaptureStartHost() / sr : 0.0;
        const double t0 = off + viewStart / sr;
        const double t1 = off + (viewStart + viewSpan) / sr;
        const int decimals = step < 0.05 ? 3 : (step < 1.0 ? 2 : 0);

        g.setFont (11.0f);
        for (double t = std::ceil (t0 / step) * step; t <= t1; t += step)
        {
            const float x = (float) ((((t - off) * sr) - viewStart) / viewSpan * (double) getWidth());
            g.setColour (juce::Colour (0xff6b7480));
            g.drawVerticalLine ((int) x, (float) r.getBottom() - 6.0f, (float) r.getBottom());
            g.setColour (juce::Colour (0xffaab2bd));
            g.drawText (formatTime (t, decimals), (int) x + 3, r.getY(), 80, r.getHeight(),
                        juce::Justification::centredLeft, false);
        }
    }

    void rebuildCache (int w, int h)
    {
        cache = juce::Image (juce::Image::ARGB, std::max (1, w), std::max (1, h), true);
        juce::Graphics g (cache);

        const auto& cap = proc.getCapture();
        const int64_t len = proc.getCaptureLength();
        const auto map = proc.getGainMapCopy();
        const bool analyzed = map.count > 0;

        const int64_t fade = std::min<int64_t> (
            map.minLen,
            (int64_t) std::llround ((double) proc.getFadeMs() * 0.001 * proc.getSampleRate()));

        const double spp = viewSpan / (double) w;
        const float mid   = (float) h * 0.5f;
        const float scale = mid * 0.92f;

        for (int x = 0; x < w; ++x)
        {
            const int64_t s0 = (int64_t) std::floor (viewStart + (double) x * spp);
            int64_t s1 = (int64_t) std::floor (viewStart + (double) (x + 1) * spp);
            if (s1 <= s0) s1 = s0 + 1;
            if (s0 >= len) break;
            if (s0 < 0) continue;
            if (s1 > len) s1 = len;

            const int64_t stride = std::max<int64_t> (1, (s1 - s0) / 128);
            float mn = cap[(size_t) s0], mx = mn;
            for (int64_t j = s0; j < s1; j += stride)
            {
                const float v = cap[(size_t) j];
                mn = std::min (mn, v);
                mx = std::max (mx, v);
            }

            if (analyzed)
            {
                // original (tenue)
                const float top = mid - juce::jlimit (-1.0f, 1.0f, mx) * scale;
                const float bot = mid - juce::jlimit (-1.0f, 1.0f, mn) * scale;
                g.setColour (juce::Colour (0xff3b4350));
                g.fillRect ((float) x, top, 1.0f, std::max (1.0f, bot - top));
            }

            const float gain = analyzed ? gainAt (map, (s0 + s1) / 2, fade) : 1.0f;
            const float top = mid - juce::jlimit (-1.0f, 1.0f, mx * gain) * scale;
            const float bot = mid - juce::jlimit (-1.0f, 1.0f, mn * gain) * scale;
            g.setColour (analyzed ? juce::Colour (0xff4cc9f0) : juce::Colour (0xff9fb3c8));
            g.fillRect ((float) x, top, 1.0f, std::max (1.0f, bot - top));
        }

        cacheDirty = false;
    }

    //==========================================================================
    NivelPluginProcessor& proc;
    juce::ScrollBar scrollBar { false };   // barra horizontal
    juce::Image cache;
    bool cacheDirty = true;
    bool autoFit = true;
    bool follow = true;

    double viewStart = 0.0;
    double viewSpan  = 96000.0;
    int lastDragX = 0;
    int dragCut = -1;
    bool dragPushed = false;
    bool draggingCursor = false;
    int64_t cursorIdx = -1;

    uint32_t lastRev = 0xffffffffu;
    int64_t  lastLen = -1;
    float    lastFade = -1.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (WaveformView)
};

//==============================================================================
// Cuadro de confirmacion propio (sin depender de dialogos del sistema)
class ConfirmOverlay : public juce::Component
{
public:
    ConfirmOverlay()
    {
        yesBtn.setButtonText (T ("Sí, limpiar"));
        noBtn.setButtonText ("Cancelar");
        addAndMakeVisible (yesBtn);
        addAndMakeVisible (noBtn);

        yesBtn.onClick = [this] { setVisible (false); if (onYes) onYes(); };
        noBtn.onClick  = [this] { setVisible (false); };
    }

    std::function<void()> onYes;

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colours::black.withAlpha (0.6f));

        auto box = boxBounds();
        g.setColour (juce::Colour (0xff252a31));
        g.fillRoundedRectangle (box.toFloat(), 8.0f);
        g.setColour (juce::Colour (0xff4a5360));
        g.drawRoundedRectangle (box.toFloat(), 8.0f, 1.0f);

        auto title = box.removeFromTop (46);
        g.setColour (juce::Colours::white);
        g.setFont (18.0f);
        g.drawText (T ("¿Estás seguro?"), title, juce::Justification::centred);

        box.removeFromBottom (44);
        g.setColour (juce::Colour (0xffaab2bd));
        g.setFont (13.0f);
        g.drawFittedText (T ("Se borrarán la captura, los cortes y el análisis.\nEsta acción no se puede deshacer."),
                          box.reduced (14, 0), juce::Justification::centred, 3);
    }

    void resized() override
    {
        auto box = boxBounds();
        auto btns = box.removeFromBottom (44).reduced (16, 8);
        yesBtn.setBounds (btns.removeFromLeft (btns.getWidth() / 2 - 6));
        btns.removeFromLeft (12);
        noBtn.setBounds (btns);
    }

    void mouseDown (const juce::MouseEvent&) override {}   // bloquea lo que hay debajo

private:
    juce::Rectangle<int> boxBounds() const
    {
        return juce::Rectangle<int> (0, 0, 340, 150).withCentre (getLocalBounds().getCentre());
    }

    juce::TextButton yesBtn, noBtn;
};

//==============================================================================
class NivelPluginEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    using SliderAtt = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAtt = juce::AudioProcessorValueTreeState::ButtonAttachment;

    explicit NivelPluginEditor (NivelPluginProcessor& p)
        : juce::AudioProcessorEditor (&p), proc (p), wave (p)
    {
        setLookAndFeel (&lnf);
        lnf.setColourScheme (juce::LookAndFeel_V4::getMidnightColourScheme());

        captureBtn.setButtonText (T ("Capturar Audio"));
        analyzeBtn.setButtonText ("Analizar");
        splitBtn.setButtonText (T ("Dividir Automáticamente"));
        clearBtn.setButtonText ("Limpiar");
        undoBtn.setButtonText ("Deshacer");
        redoBtn.setButtonText ("Rehacer");
        zoomOutBtn.setButtonText ("Zoom -");
        zoomInBtn.setButtonText ("Zoom +");
        fitBtn.setButtonText ("Ajustar");
        applyBtn.setButtonText ("Aplicar ganancia");
        zeroBtn.setButtonText ("Cortes en cruce por cero");
        followBtn.setButtonText ("Seguir playhead");
        autoStopBtn.setButtonText (T ("Detener captura al parar el host"));

        for (auto* b : { &captureBtn, &analyzeBtn, &splitBtn, &clearBtn, &undoBtn, &redoBtn,
                         &zoomOutBtn, &zoomInBtn, &fitBtn })
            addAndMakeVisible (*b);
        addAndMakeVisible (applyBtn);
        addAndMakeVisible (zeroBtn);
        addAndMakeVisible (followBtn);
        addAndMakeVisible (autoStopBtn);

        followBtn.setToggleState (true, juce::dontSendNotification);

        auto setupSlider = [this] (juce::Slider& s, const juce::String& suffix)
        {
            s.setSliderStyle (juce::Slider::LinearHorizontal);
            s.setTextBoxStyle (juce::Slider::TextBoxRight, false, 78, 20);
            s.setTextValueSuffix (suffix);
            addAndMakeVisible (s);
        };
        auto setupLabel = [this] (juce::Label& l, const juce::String& text)
        {
            l.setText (text, juce::dontSendNotification);
            l.setJustificationType (juce::Justification::centredLeft);
            addAndMakeVisible (l);
        };

        setupSlider (targetSlider, " dBFS");
        setupSlider (fadeSlider,   " ms");
        setupSlider (threshSlider, " dBFS");
        setupSlider (minSilSlider, " ms");

        setupLabel (targetLabel, "Nivel objetivo");
        setupLabel (fadeLabel,   "Crossfade");
        setupLabel (threshLabel, "Umbral silencio");
        setupLabel (minSilLabel, T ("Silencio mín."));

        addAndMakeVisible (wave);

        targetAtt   = std::make_unique<SliderAtt> (proc.apvts, "target",    targetSlider);
        fadeAtt     = std::make_unique<SliderAtt> (proc.apvts, "fade",      fadeSlider);
        threshAtt   = std::make_unique<SliderAtt> (proc.apvts, "silthresh", threshSlider);
        minSilAtt   = std::make_unique<SliderAtt> (proc.apvts, "minsil",    minSilSlider);
        applyAtt    = std::make_unique<ButtonAtt> (proc.apvts, "apply",     applyBtn);
        zeroAtt     = std::make_unique<ButtonAtt> (proc.apvts, "zerocross", zeroBtn);
        autoStopAtt = std::make_unique<ButtonAtt> (proc.apvts, "autostop",  autoStopBtn);

        captureBtn.onClick = [this]
        {
            if (proc.isCapturing())
                proc.stopCapture();
            else
            {
                proc.startCapture();
                wave.fitView();
            }
        };
        analyzeBtn.onClick = [this] { proc.analyze(); };
        splitBtn.onClick   = [this] { proc.autoSplit(); };
        undoBtn.onClick    = [this] { proc.undo(); };
        redoBtn.onClick    = [this] { proc.redo(); };
        zoomInBtn.onClick  = [this] { wave.zoomIn(); };
        zoomOutBtn.onClick = [this] { wave.zoomOut(); };
        fitBtn.onClick     = [this] { wave.fitView(); };
        followBtn.onClick  = [this] { wave.setFollow (followBtn.getToggleState()); };

        clearBtn.onClick = [this]
        {
            if (proc.getCaptureLength() > 0 || proc.isAnalyzed())
            {
                confirm.setVisible (true);
                confirm.toFront (false);
            }
            else
            {
                proc.clearAll();
                wave.fitView();
            }
        };

        confirm.onYes = [this]
        {
            proc.clearAll();
            wave.fitView();
        };
        addChildComponent (confirm);   // siempre el ultimo: queda encima de todo

        setResizable (true, true);
        setResizeLimits (560, 300, 2400, 1200);
        setSize (800, 400);

        startTimerHz (10);
    }

    ~NivelPluginEditor() override
    {
        stopTimer();
        setLookAndFeel (nullptr);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff101317));
    }

    void resized() override
    {
        auto r = getLocalBounds().reduced (8);

        auto row1 = r.removeFromTop (28);  r.removeFromTop (4);
        auto row2 = r.removeFromTop (24);  r.removeFromTop (4);
        auto row3 = r.removeFromTop (24);  r.removeFromTop (4);
        auto row4 = r.removeFromTop (24);  r.removeFromTop (6);

        layoutRow (row1, { { &captureBtn, 1.3f }, { &analyzeBtn, 0.9f }, { &splitBtn, 1.8f },
                           { &clearBtn, 0.9f }, { &undoBtn, 1.0f }, { &redoBtn, 1.0f } }, 6);

        {
            applyBtn.setBounds (row2.removeFromRight (150));
            row2.removeFromRight (8);
            auto left = row2.removeFromLeft (row2.getWidth() / 2 - 6);
            row2.removeFromLeft (12);
            placeLabelled (left, targetLabel, targetSlider, 100);
            placeLabelled (row2, fadeLabel, fadeSlider, 100);
        }
        {
            const int third = (row3.getWidth() - 24) / 3;
            auto a = row3.removeFromLeft (third);  row3.removeFromLeft (12);
            auto b = row3.removeFromLeft (third);  row3.removeFromLeft (12);
            placeLabelled (a, threshLabel, threshSlider, 100);
            placeLabelled (b, minSilLabel, minSilSlider, 100);
            zeroBtn.setBounds (row3);
        }

        layoutRow (row4, { { &zoomOutBtn, 0.8f }, { &zoomInBtn, 0.8f }, { &fitBtn, 0.9f },
                           { &followBtn, 1.4f }, { &autoStopBtn, 2.2f } }, 6);

        wave.setBounds (r);
        confirm.setBounds (getLocalBounds());
    }

private:
    static void layoutRow (juce::Rectangle<int> row,
                           std::initializer_list<std::pair<juce::Component*, float>> items,
                           int gap)
    {
        float total = 0.0f;
        for (const auto& it : items)
            total += it.second;

        const float avail = (float) (row.getWidth() - gap * ((int) items.size() - 1));
        for (const auto& it : items)
        {
            const int w = juce::roundToInt (avail * it.second / total);
            it.first->setBounds (row.removeFromLeft (w));
            row.removeFromLeft (gap);
        }
    }

    static void placeLabelled (juce::Rectangle<int> cell, juce::Label& l, juce::Slider& s, int labelW)
    {
        l.setBounds (cell.removeFromLeft (labelW));
        s.setBounds (cell);
    }

    void timerCallback() override
    {
        const bool capturing = proc.isCapturing();
        const auto txt = capturing ? T ("Detener captura") : T ("Capturar Audio");
        if (captureBtn.getButtonText() != txt)
            captureBtn.setButtonText (txt);

        captureBtn.setColour (juce::TextButton::buttonColourId,
                              capturing ? juce::Colour (0xff9b2226)
                                        : lnf.findColour (juce::TextButton::buttonColourId));

        const bool hasData = proc.isBufferValid() && proc.getCaptureLength() > 0;
        analyzeBtn.setEnabled (hasData);
        splitBtn.setEnabled (hasData);
        undoBtn.setEnabled (proc.canUndo());
        redoBtn.setEnabled (proc.canRedo());
    }

    juce::LookAndFeel_V4 lnf;
    NivelPluginProcessor& proc;

    juce::TextButton captureBtn, analyzeBtn, splitBtn, clearBtn, undoBtn, redoBtn,
                     zoomOutBtn, zoomInBtn, fitBtn;
    juce::ToggleButton applyBtn, zeroBtn, followBtn, autoStopBtn;
    juce::Slider targetSlider, fadeSlider, threshSlider, minSilSlider;
    juce::Label targetLabel, fadeLabel, threshLabel, minSilLabel;
    WaveformView wave;
    ConfirmOverlay confirm;

    std::unique_ptr<SliderAtt> targetAtt, fadeAtt, threshAtt, minSilAtt;
    std::unique_ptr<ButtonAtt> applyAtt, zeroAtt, autoStopAtt;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (NivelPluginEditor)
};

//==============================================================================
juce::AudioProcessorEditor* NivelPluginProcessor::createEditor()
{
    return new NivelPluginEditor (*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NivelPluginProcessor();
}
