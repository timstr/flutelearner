#include <SFML/Audio.hpp>
#include <neuralnetwork.hpp>
#include <fft.hpp>

#include <iostream>
#include <map>
#include <thread>

constexpr std::size_t nextPowerOf2(std::size_t n){
    // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;
    return n;
}

// The audio sampling rate, in Hz
const std::size_t sampleRate = 48000;

// the length of each note, in samples (4 notes per second)
const std::size_t noteLength = sampleRate / 4;

// The length of each note, rounded up to the nearest power of 2
const std::size_t noteLengthPadded = nextPowerOf2(noteLength);

// the number of frequencies that are considered for each note
const std::size_t numFrequences = noteLengthPadded / 2;

// the number of distinct notes that appear in the input sound.
//This corresponds to 3 octaves plus a special "silence" note
const std::size_t numNotes = 12 * 3 + 1;

// the names of all notes in the order they appear in the training data
// (for convenience and for printing)
const std::array<std::string, numNotes> noteNames = {    "(silence)",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
};

// The sound containing both the training data and the test data
sf::SoundBuffer& getInputSound(){
    static sf::SoundBuffer buff;
    static bool init;
    if (!init){
        buff.loadFromFile("../sound/flute.wav");
        init = true;
    }
    return buff;
}

// The number of training examples in the input file.
// This corresponds to the first 30 seconds of audio
// at four notes per second
const std::size_t numTrainingExamples = 30 * 4;

// Hand-written labels for the training data
// Note that 0 means silence
const std::array<std::size_t, numTrainingExamples> trainingLabels = {
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    1,  0,  2,  0,  3,  0,  4,  0,  5,  0,  6,  0,
    7,  0,  8,  0,  9,  0,  10, 0,  11, 0,  12, 0,
    13, 0,  14, 0,  15, 0,  16, 0,  17, 0,  18, 0,
    19, 0,  20, 0,  21, 0,  22, 0,  23, 0,  24, 0,
    25, 0,  26, 0,  27, 0,  28, 0,  29, 0,  30, 0,
    31, 0,  32, 0,  33, 0,  34, 0,  35, 0,  36, 0
};

// Given the index of a note in the input sound, reads the
// sound at that index, computes the fft, and returns the
// magnitudes of the positive frequency components
std::vector<double> getNoteFrequences(size_t noteIndex){
    const std::size_t startSample = noteIndex * noteLength;
    if (getInputSound().getChannelCount() != 1){
        throw std::runtime_error("I though the input was supposed to be mono audio :(");
    }
    const auto samples = getInputSound().getSamples();

    std::vector<std::complex<double>> x;
    x.reserve(noteLengthPadded);
    for (std::size_t i = 0; i < noteLength; ++i){
        const auto s = samples[startSample + i];
        x.push_back(static_cast<double>(s) / static_cast<double>(std::numeric_limits<sf::Int16>::max()));
    }

    for (std::size_t i = noteLength; i < noteLengthPadded; ++i){
        x.push_back({});
    }

    if (x.size() != noteLengthPadded){
        throw std::runtime_error("Oops");
    }

    fft(x);

    std::vector<double> output;
    output.reserve(noteLengthPadded / 2);
    for (std::size_t i = 0; i < noteLengthPadded / 2; ++i){
        output.push_back(std::abs(x[i]));
    }

    return output;
}

using NetworkType = NeuralNetwork<numNotes, 256, 256, numFrequences>;

auto network = NetworkType{};

void train(){
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> outputs;

    // generate training data
    for (std::size_t i = 0; i < numTrainingExamples; ++i){
        inputs.push_back(getNoteFrequences(i));

        std::vector<double> y(numNotes, 0.0);
        y[trainingLabels[i]] = 1.0;

        outputs.push_back(std::move(y));
    }

    // zip input/output pairs together
    std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> trainingData;
    for (std::size_t i = 0; i < numTrainingExamples; ++i){
        auto x = NetworkType::InputType{inputs[i]};
        auto y = NetworkType::OutputType{outputs[i]};
        trainingData.push_back({x, y});
    }

    // restore from last training session
    network.loadWeights("weights 5.dat");

    // train!!!
    auto loss = 1e6;
    while (loss > 1.0){
        loss = network.takeStep(trainingData, 10.0, 0.5);
        std::cout << "loss = " << loss << '\n';
    }
    network.saveWeights("weights 1.dat");
    while (loss > 0.5){
        loss = network.takeStep(trainingData, 10.0, 0.5);
        std::cout << "loss = " << loss << '\n';
    }
    network.saveWeights("weights 0.5.dat");
    while (loss > 0.1){
        loss = network.takeStep(trainingData, 10.0, 0.5);
        std::cout << "loss = " << loss << '\n';
    }
    network.saveWeights("weights 0.1.dat");
    while (loss > 0.05){
        loss = network.takeStep(trainingData, 10.0, 0.5);
        std::cout << "loss = " << loss << '\n';
    }
    network.saveWeights("weights 0.05.dat");
}

int main(){

    network.randomizeWeights();

    train();

    return 0;
}