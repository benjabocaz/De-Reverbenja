#pragma once
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_dsp/juce_dsp.h>
#include <array>
#include <vector>

class SimpleDeReverbProcessor : public juce::AudioProcessor
{
public:
    SimpleDeReverbProcessor();
    ~SimpleDeReverbProcessor() override = default;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    static constexpr int fftOrder = 10;                  // 1024 muestras (~21ms)
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int hopSize = fftSize / 4;          // 75% overlap
    static constexpr int numBins = fftSize / 2 + 1;      // 513 bandas

    struct ChannelState
    {
        std::array<float, fftSize> inputBuffer{};
        std::array<float, fftSize> outputAccum{};
        std::array<float, hopSize> outputReady{};
        std::array<float, fftSize * 2> fftData{};

        std::array<float, numBins> prevPower{};
        std::array<float, numBins> reverbEst{};
        std::array<float, numBins> rawGain{};
        std::array<float, numBins> smoothGain{};

        int hopIndex = 0;

        void reset()
        {
            inputBuffer.fill(0.0f);
            outputAccum.fill(0.0f);
            outputReady.fill(0.0f);
            fftData.fill(0.0f);
            prevPower.fill(0.0f);
            reverbEst.fill(0.0f);
            rawGain.fill(1.0f);
            smoothGain.fill(1.0f);
            hopIndex = 0;
        }
    };

    void processFrame(ChannelState& state, float amount, float roomDecay, float roomFocus, float smoothness);

    juce::dsp::FFT fft { fftOrder };
    juce::dsp::WindowingFunction<float> window { fftSize, juce::dsp::WindowingFunction<float>::hann, false };
    std::vector<ChannelState> channels;
    double currentSampleRate = 44100.0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimpleDeReverbProcessor)
};
