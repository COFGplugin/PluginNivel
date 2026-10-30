#include <JuceHeader.h>
#include <vector>

// =====================================================================
// 1. LÓGICA DE AUDIO (PROCESADOR)
// =====================================================================
class NivelPluginAudioProcessor : public juce::AudioProcessor
{
public:
    NivelPluginAudioProcessor() : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::mono(), true)
        .withOutput("Output", juce::AudioChannelSet::mono(), true)) {}

    ~NivelPluginAudioProcessor() override {}

    void prepareToPlay(double sampleRate, int samplesPerBlock) override
    {
        bufferInterno.setSize(1, (int)(sampleRate * 300));
        bufferInterno.clear();
        posicionEscritura = 0;
        waveformData.clear();
        grabandoBuffer = false;
    }

    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override { return true; }
    
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
    {
        juce::ScopedNoDenormals noDenormals;
        if (grabandoBuffer && posicionEscritura + buffer.getNumSamples() < bufferInterno.getNumSamples())
        {
            bufferInterno.copyFrom(0, posicionEscritura, buffer, 0, 0, buffer.getNumSamples());
            posicionEscritura += buffer.getNumSamples();
        }
    }

    void analizarAudio()
    {
        waveformData.clear();
        int numSamples = posicionEscritura;
        if (numSamples == 0) return;

        // TU LÓGICA EXACTA: Buscar el pico máximo
        float picoMaximo = 0.0f;
        for (int i = 0; i < numSamples; ++i)
        {
            float muestra = std::abs(bufferInterno.getSample(0, i));
            if (muestra > picoMaximo) picoMaximo = muestra;
        }

        if (picoMaximo > 0.0f)
        {
            // Calcular dB y aplicar ganancia
            float picoDBFS = juce::Decibels::gainToDecibels(picoMaximo);
            float gananciaDB = nivelObjetivoDB - picoDBFS;
            float gananciaLineal = juce::Decibels::decibelsToGain(gananciaDB);
            bufferInterno.applyGain(0, 0, numSamples, gananciaLineal);
        }

        // Preparar datos para la onda visual
        for (int i = 0; i < 800; ++i)
        {
            int index = (i * numSamples) / 800;
            waveformData.push_back(bufferInterno.getSample(0, index));
        }
    }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "NivelPlugin"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int index) override {}
    const juce::String getProgramName(int index) override { return {}; }
    void changeProgramName(int index, const juce::String& newName) override {}
    void getStateInformation(juce::MemoryBlock& destData) override {}
    void setStateInformation(const void* data, int sizeInBytes) override {}

    std::vector<float> waveformData;
    float nivelObjetivoDB = -18.0f;
    bool grabandoBuffer = false;
    juce::AudioBuffer<float> bufferInterno;
    int posicionEscritura = 0;
};

// =====================================================================
// 2. INTERFAZ GRÁFICA (EDITOR)
// =====================================================================
class NivelPluginAudioProcessorEditor : public juce::AudioProcessorEditor, public juce::Timer
{
public:
    NivelPluginAudioProcessorEditor(NivelPluginAudioProcessor& p)
        : AudioProcessorEditor(&p), audioProcessor(p)
    {
        setSize(800, 400); // Ventana del tamaño que pediste

        addAndMakeVisible(botonGrabar);
        botonGrabar.onClick = [this]() {
            audioProcessor.grabandoBuffer = !audioProcessor.grabandoBuffer;
            botonGrabar.setButtonText(audioProcessor.grabandoBuffer ? "Detener Captura" : "Capturar Audio");
            if (!audioProcessor.grabandoBuffer) {
                audioProcessor.analizarAudio();
                repaint();
            }
        };

        addAndMakeVisible(botonAnalizar);
        botonAnalizar.onClick = [this]() {
            audioProcessor.analizarAudio();
            repaint();
        };

        addAndMakeVisible(sliderObjetivo);
        sliderObjetivo.setRange(-40.0, 0.0, 0.1);
        sliderObjetivo.setValue(-18.0);
        sliderObjetivo.setTextBoxStyle(juce::Slider::TextBoxRight, false, 60, 20);
        
        addAndMakeVisible(labelObjetivo);
        labelObjetivo.setText("Nivel Objetivo (dBFS):", juce::dontSendNotification);

        startTimerHz(30);
    }

    ~NivelPluginAudioProcessorEditor() override {}

    void paint(juce::Graphics& g) override
    {
        g.fillAll(juce::Colours::black);

        auto bounds = getLocalBounds().reduced(10);
        bounds.removeFromTop(50);

        g.setColour(juce::Colours::cyan);
        if (audioProcessor.waveformData.size() > 0)
        {
            juce::Path path;
            path.startNewSubPath(bounds.getX(), bounds.getCentreY());
            for (size_t i = 0; i < audioProcessor.waveformData.size(); ++i)
            {
                float x = bounds.getX() + (i * bounds.getWidth() / audioProcessor.waveformData.size());
                float y = bounds.getCentreY() - (audioProcessor.waveformData[i] * bounds.getHeight() / 2.0f);
                path.lineTo(x, y);
            }
            g.strokePath(path, juce::PathStrokeType(1.5f));
        }
    }

    void resized() override
    {
        auto area = getLocalBounds().reduced(10);
        auto topBar = area.removeFromTop(30);

        botonGrabar.setBounds(topBar.removeFromLeft(120));
        topBar.removeFromLeft(10);
        botonAnalizar.setBounds(topBar.removeFromLeft(100));
        topBar.removeFromLeft(20);
        labelObjetivo.setBounds(topBar.removeFromLeft(150));
        sliderObjetivo.setBounds(topBar.removeFromLeft(200));
    }

    void timerCallback() override { repaint(); }

private:
    NivelPluginAudioProcessor& audioProcessor;
    juce::TextButton botonGrabar { "Capturar Audio" };
    juce::TextButton botonAnalizar { "Analizar" };
    juce::Slider sliderObjetivo;
    juce::Label labelObjetivo;
};

juce::AudioProcessorEditor* NivelPluginAudioProcessor::createEditor()
{
    return new NivelPluginAudioProcessorEditor(*this);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new NivelPluginAudioProcessor();
}
