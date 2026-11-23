#include "PluginProcessor.h"

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
        juce::ParameterID{"amount", 1}, "De-Reverb Amount",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f), 0.5f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"sensitivity", 1}, "Sensitivity",
        juce::NormalisableRange<float>(0.1f, 2.0f, 0.01f), 1.0f));

    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{"release", 1}, "Tail Release (ms)",
        juce::NormalisableRange<float>(20.0f, 400.0f, 1.0f), 120.0f));

    return { params.begin(), params.end() };
}

void SimpleDeReverbProcessor::prepareToPlay(double sampleRate, int)
{
    currentSampleRate = sampleRate;
    fastEnv = 0.0f;
    slowEnv = 0.0f;
    smoothGain = 1.0f;
}

bool SimpleDeReverbProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    return layouts.getMainOutputChannelSet() == juce::AudioChannelSet::mono()
        || layouts.getMainOutputChannelSet() == juce::AudioChannelSet::stereo();
}

void SimpleDeReverbProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;

    const float amount = apvts.getRawParameterValue("amount")->load();
    const float sensitivity = apvts.getRawParameterValue("sensitivity")->load();
    const float releaseMs = apvts.getRawParameterValue("release")->load();

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    const float fastAttack = std::exp(-1.0f / (static_cast<float>(currentSampleRate) * 0.003f));
    const float fastRelease = std::exp(-1.0f / (static_cast<float>(currentSampleRate) * 0.040f));
    const float slowAttack = std::exp(-1.0f / (static_cast<float>(currentSampleRate) * 0.050f));
    const float slowRelease = std::exp(-1.0f / (static_cast<float>(currentSampleRate) * (releaseMs * 0.001f)));
    const float gainSmoothCoeff = std::exp(-1.0f / (static_cast<float>(currentSampleRate) * 0.015f));

    for (int i = 0; i < numSamples; ++i)
    {
        float peak = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch)
            peak = std::max(peak, std::abs(buffer.getSample(ch, i)));

        fastEnv = (peak > fastEnv)
            ? fastAttack * fastEnv + (1.0f - fastAttack) * peak
            : fastRelease * fastEnv + (1.0f - fastRelease) * peak;

        slowEnv = (peak > slowEnv)
            ? slowAttack * slowEnv + (1.0f - slowAttack) * peak
            : slowRelease * slowEnv + (1.0f - slowRelease) * peak;

        float ratio = fastEnv / (slowEnv + 1e-6f);
        float targetGain = 1.0f;

        if (ratio < sensitivity)
        {
            float reduction = juce::jlimit(0.0f, 1.0f, ratio / sensitivity);
            targetGain = (1.0f - amount) + (amount * reduction * reduction);
        }

        smoothGain = gainSmoothCoeff * smoothGain + (1.0f - gainSmoothCoeff) * targetGain;

        for (int ch = 0; ch < numChannels; ++ch)
            buffer.setSample(ch, i, buffer.getSample(ch, i) * smoothGain);
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
