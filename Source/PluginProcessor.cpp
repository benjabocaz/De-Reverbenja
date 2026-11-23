#include "PluginProcessor.h"
#include <algorithm>
#include <cmath>

SimpleDeReverbProcessor::SimpleDeReverbProcessor()
    : AudioProcessor(BusesProperties()
        .withInput("Input", juce::AudioChannelSet::stereo(), true)
        .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
{
}

juce::AudioProcessorValueTreeState::ParameterLayout SimpleDeReverbProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"amount", 1}, "De-Room Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.65f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"focus", 1}, "Ambiance Focus",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.50f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"decay", 1}, "Room Decay",
        juce::NormalisableRange<float>(0.05f, 0.95f, 0.01f), 0.55f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"smooth", 1}, "Smoothness",
        juce::NormalisableRange<float>(0.0f, 0.95f, 0.01f), 0.50f));

    return { params.begin(), params.end() };
}

void SimpleDeReverbProcessor::prepareToPlay(double sampleRate, int)
{
    currentSampleRate = sampleRate;
    setLatencySamples(fftSize); // Compensación automática de latencia en Logic Pro

    channels.resize(static_cast<size_t>(std::max(1, getTotalNumInputChannels())));
    for (auto& ch : channels)
        ch.reset();
}

bool SimpleDeReverbProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void SimpleDeReverbProcessor::processFrame(ChannelState& state, float amount, float roomDecay, float roomFocus, float smoothness)
{
    // 1. Copiar ventana actual y aplicar Hann
    std::copy(state.inputBuffer.begin(), state.inputBuffer.end(), state.fftData.begin());
    std::fill(state.fftData.begin() + fftSize, state.fftData.end(), 0.0f);
    window.multiplyWithWindowingTable(state.fftData.data(), fftSize);

    // 2. Pasar al dominio de la frecuencia (FFT)
    fft.performRealOnlyForwardTransform(state.fftData.data(), true);

    // Factor de agresividad para atrapar reflexiones tempranas y tono de sala
    const float overSub = 0.6f + roomFocus * 2.4f;
    const float minGain = 1.0f - (amount * 0.94f);
    const float timeSmooth = 0.25f + smoothness * 0.65f;

    // 3. Estimar energía de sala por cada banda de frecuencia
    for (int k = 0; k < numBins; ++k)
    {
        const float re = state.fftData[2 * k];
        const float im = state.fftData[2 * k + 1];
        const float power = re * re + im * im;

        // Usamos la energía retrasada (prevPower) para que el sonido directo pase intacto
        // mientras las reflexiones de la sala quedan atrapadas en reverbEst
        state.reverbEst[k] = std::max(roomDecay * state.reverbEst[k],
                                      0.35f * state.reverbEst[k] + 0.65f * state.prevPower[k]);
        state.prevPower[k] = power;

        const float estRoom = state.reverbEst[k] * overSub;
        const float cleanPower = std::max(0.0f, power - estRoom);
        const float wiener = cleanPower / (power + 1e-9f);

        state.rawGain[k] = juce::jlimit(minGain, 1.0f, (1.0f - amount) + amount * wiener);
    }

    // 4. Suavizado espectral (evita sonido "metálico" o robótico) y aplicación de ganancia
    for (int k = 0; k < numBins; ++k)
    {
        float g = state.rawGain[k];
        if (k > 0 && k < numBins - 1)
            g = 0.25f * state.rawGain[k - 1] + 0.50f * state.rawGain[k] + 0.25f * state.rawGain[k + 1];

        state.smoothGain[k] = timeSmooth * state.smoothGain[k] + (1.0f - timeSmooth) * g;

        state.fftData[2 * k]     *= state.smoothGain[k];
        state.fftData[2 * k + 1] *= state.smoothGain[k];
    }

    // 5. Volver al dominio del tiempo (IFFT) y reconstruir con Overlap-Add
    fft.performRealOnlyInverseTransform(state.fftData.data());
    window.multiplyWithWindowingTable(state.fftData.data(), fftSize);

    constexpr float normFactor = 2.0f / 3.0f; // Compensación exacta para Hann con 75% overlap
    for (int i = 0; i < fftSize; ++i)
        state.outputAccum[i] += state.fftData[i] * normFactor;

    std::copy(state.outputAccum.begin(), state.outputAccum.begin() + hopSize, state.outputReady.begin());
    std::copy(state.outputAccum.begin() + hopSize, state.outputAccum.end(), state.outputAccum.begin());
    std::fill(state.outputAccum.end() - hopSize, state.outputAccum.end(), 0.0f);
    std::copy(state.inputBuffer.begin() + hopSize, state.inputBuffer.end(), state.inputBuffer.begin());
}

void SimpleDeReverbProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const float amount = apvts.getRawParameterValue("amount")->load();
    const float focus = apvts.getRawParameterValue("focus")->load();
    const float decay = apvts.getRawParameterValue("decay")->load();
    const float smooth = apvts.getRawParameterValue("smooth")->load();

    const int numChannels = std::min(buffer.getNumChannels(), static_cast<int>(channels.size()));
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto& state = channels[static_cast<size_t>(ch)];
        float* channelData = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i)
        {
            state.inputBuffer[fftSize - hopSize + state.hopIndex] = channelData[i];
            channelData[i] = state.outputReady[state.hopIndex];
            state.hopIndex++;

            if (state.hopIndex == hopSize)
            {
                state.hopIndex = 0;
                processFrame(state, amount, decay, focus, smooth);
            }
        }
    }
}

juce::AudioProcessorEditor* SimpleDeReverbProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void SimpleDeReverbProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void SimpleDeReverbProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState != nullptr && xmlState->hasTagName(apvts.state.getType()))
        apvts.replaceState(juce::ValueTree::fromXml(*xmlState));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SimpleDeReverbProcessor();
}
