#include <SFML/Audio.hpp>
#include <neuralnetwork.hpp>
#include <fft.hpp>

#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>

std::random_device randDev;
std::default_random_engine randEng{randDev()};

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

// The length of each note, rounded down to the nearest power of 2
const std::size_t noteLengthTruncated = nextPowerOf2(noteLength) / 2;

// the number of frequencies that are considered for each note
// half as many because negative frequencies are discarded,
// and 1/8 as many as that because all frequencies above 1500 Hz
const std::size_t numFrequences = noteLengthTruncated / 2 / 8;

// the number of distinct notes that appear in the input sound.
//This corresponds to 3 octaves plus a special "silence" note
const std::size_t numNotes = 12 * 3 + 1;

// the names of all notes in the order they appear in the training data
// (for convenience and for printing)
const std::array<std::string, numNotes> noteNames = {
    "(silence)",
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5", "C#5", "D5", "D#5", "E5", "F5", "F#5", "G5", "G#5", "A5", "A#5", "B5"
};

// The number of training examples in the input file.
// This corresponds to the first 30 seconds of audio
// at four notes per second
// const std::size_t numTrainingExamples = 30 * 4;

// Hand-written (note index, label) pairs for the training data
// Note that a label of 0 denotes silence
const auto trainingExamples = std::vector<std::pair<std::size_t, std::size_t>>{
    // First batch of training data: one note every quarter second
    {0,  1 }, {1,  2 }, {2,  3 }, {3,  4 }, {4,  5 }, {5,  6 },
    {6,  7 }, {7,  8 }, {8,  9 }, {9,  10}, {10, 11}, {11, 12},
    {12, 13}, {13, 14}, {14, 15}, {15, 16}, {16, 17}, {17, 18},
    {18, 19}, {19, 20}, {20, 21}, {21, 22}, {22, 23}, {23, 24},
    {24, 25}, {25, 26}, {26, 27}, {27, 28}, {28, 29}, {29, 30},
    {30, 31}, {31, 32}, {32, 33}, {33, 34}, {34, 35}, {35, 36},
        
    // Maybe learn some silence?
    {37, 0 }, {38, 0 }, {39, 0}, {40, 0}, {41, 0 }, {42, 0 }, {43, 0}, {44, 0},
    
    // Second batch of training data: one note every half second,
    // starting at 12 seconds (48 quarter seconds)
    {48,  1 }, {50,  2 }, {52,  3 }, {54,  4 }, {56,  5 }, {58,  6 },
    {60,  7 }, {62,  8 }, {64,  9 }, {66,  10}, {68,  11}, {70,  12},
    {72,  13}, {74,  14}, {76,  15}, {78,  16}, {80,  17}, {82,  18},
    {84,  19}, {86,  20}, {88,  21}, {90,  22}, {92,  23}, {94,  24},
    {96,  25}, {98,  26}, {100, 27}, {102, 28}, {104, 29}, {106, 30},
    {108, 31}, {110, 32}, {112, 33}, {114, 34}, {116, 35}, {118, 36},
};

// Given the index of a note in the input sound, reads the
// sound at that index, computes the fft, and returns the
// magnitudes of the positive frequency components
std::vector<double> getNoteFrequences(const sf::SoundBuffer& buff, std::size_t noteIndex, std::size_t samplesOffset, double noiseAmplitude){
    const std::size_t startSample = noteIndex * noteLength + samplesOffset;
    if (buff.getChannelCount() != 1){
        throw std::runtime_error("I though the input was supposed to be mono audio :(");
    }
    const auto samples = buff.getSamples();

    std::vector<std::complex<double>> x;
    x.reserve(noteLengthTruncated);
    const auto k = 1.0 / static_cast<double>(std::numeric_limits<sf::Int16>::max());
    const auto dist = std::uniform_real_distribution<double>{-1.0, 1.0};
    for (std::size_t i = 0; i < noteLengthTruncated; ++i){
        const auto s = samples[startSample + i];
        const auto win = 0.5 * (1.0 - std::cos(static_cast<double>(i) * pi / static_cast<double>(noteLengthTruncated)));
        x.push_back((static_cast<double>(s) * k + noiseAmplitude * dist(randEng)) * win);
    }



    fft(x);

    // extract lower frequencies, take logarithm of magnitude
    std::vector<double> output;
    output.reserve(numFrequences);
    for (std::size_t i = 0; i < numFrequences; ++i){
        const auto mag = std::abs(x[i]);
        output.push_back(std::log(std::max(mag, 1e-6)));
    }

    return output;
}

using NetworkType = NeuralNetwork<numNotes, 64, 128, 256, numFrequences>;

auto network = NetworkType{};

std::pair<std::size_t, double> predict(const sf::SoundBuffer& buff, std::size_t noteIndex){
    auto x = getNoteFrequences(buff, noteIndex, 0, 0.0);
    auto y = network.compute(x);
    auto maxIt = std::max_element(y.begin(), y.end());
    assert(maxIt != y.end());
    return {maxIt - y.begin(), *maxIt};
}

double trainingAccuracy(const sf::SoundBuffer& buff){
    std::size_t numCorrect = 0;
    for (const auto& inputAndLabel : trainingExamples){
        const auto [actualLabel, conf] = predict(buff, inputAndLabel.first);
        if (inputAndLabel.second == actualLabel){
            ++numCorrect;
        }
    }
    return static_cast<double>(numCorrect) / static_cast<double>(trainingExamples.size());
}

// physical storage for input and output data
std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;

void prepareTrainingData(const sf::SoundBuffer& buff){
    std::cout << "Preparing training data...";

    // number of times each traning note is repeated at a random time offset
    const std::size_t upsampling = 128;

    // maximum amplitude of random amount of white noise that is added to signal
    const double maxNoiseAmp = 0.1;
    auto noiseDist = std::uniform_real_distribution<double>{0.0, maxNoiseAmp};

    // maximum sample offset by which each training example is randomly shifted
    const std::size_t maxOffset = noteLengthTruncated / 8;
    auto offsetDist = std::uniform_int_distribution<std::size_t>{0, maxOffset};

    for (const auto& noteIndexAndLabel : trainingExamples){
        for (std::size_t i = 0; i < upsampling; ++i){
            auto input = getNoteFrequences(buff, noteIndexAndLabel.first, offsetDist(randEng), noiseDist(randEng));
            auto output = std::vector<double>(numNotes, 0.0);
            output.at(noteIndexAndLabel.second) = 1.0;
            trainingData.push_back(std::pair{std::move(input), std::move(output)});
        }
    }
    std::cout << " done.\n";
}

std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> randomTrainingBatch(std::size_t size){
    std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> ret;

    auto dist = std::uniform_int_distribution<std::size_t>{0, trainingData.size() - 1};

    for (std::size_t i = 0; i < size; ++i){
        const auto& [input, output] = trainingData.at(dist(randEng));
        ret.push_back(std::pair{NetworkType::InputType{input}, NetworkType::OutputType{output}});
    }
    
    return ret;
}

void trainNetwork(const sf::SoundBuffer& buff){
    std::atomic<double> rate = 0.05;
    std::atomic<double> momentum = 0.9;
    std::mutex networkMutex, nextUpMutex, lowPriorityMutex;


    std::condition_variable cond;

    std::atomic<bool> done = false;

    const auto trainUntil = [&](double desiredLoss){

        std::size_t count = 0;
        auto loss = 1e6;
        while (loss > desiredLoss && !done){
            for (size_t i = 0; i < 50 && !done.load(std::memory_order_relaxed); ++i){
                lowPriorityMutex.lock();
                nextUpMutex.lock();
                networkMutex.lock();
                nextUpMutex.unlock();
                loss = 0.0;
                const auto batchSize = 1 << 8;
                for (size_t i = 0; i < batchSize; ++i){
                    auto d = randomTrainingBatch(8);
                    loss += network.takeStep(
                        d,
                        rate.load(std::memory_order_relaxed),
                        momentum.load(std::memory_order_relaxed)
                    );
                    ++count;
                }
                loss /= static_cast<double>(batchSize);
                std::cout << "loss = " << loss << '\n';
                networkMutex.unlock();
                lowPriorityMutex.unlock();
            }
            std::cout << "    training accuracy = " << trainingAccuracy(buff) * 100.0 << "%\n";
            std::cout << "    total iterations so far: " << count << '\n';
        }
    };

    auto inputThread = std::thread{[&](){
        char ch = {};
        while (!done.load(std::memory_order_relaxed) && std::cin >> ch){
            nextUpMutex.lock();
            networkMutex.lock();
            nextUpMutex.unlock();
            if (ch == 's'){
                // save network
                std::cout << "\nSaving network state to a file...\nPlease enter a file name:\n";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                if (std::string line; std::getline(std::cin, line)){
                    try {
                        network.saveWeights(line);
                        std::cout << "Done.\n";
                    } catch (...) {
                        std::cout << "That's a bad file name, sorry.\n";
                    }
                } else {
                    std::cout << "Dang, that didn't work.\n";
                }
            } else if (ch == 'r'){
                // restore network
                std::cout << "\nRestoring network state from a file...\nPlease enter a file name:\n";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                if (std::string line; std::getline(std::cin, line)){
                    try {
                        network.loadWeights(line);
                        std::cout << "Done.\n";
                    } catch (...) {
                        std::cout << "That's a bad file, sorry.\n";
                    }
                } else {
                    std::cout << "Dang, that didn't work.\n";
                }
            } else if (ch == '='){
                // increase training rate
                const auto oldRate = rate.load(std::memory_order_relaxed);
                const auto newRate = oldRate * 1.1;
                std::cout << "The learning rate is now " << newRate << '\n';
                rate.store(newRate, std::memory_order_relaxed);
            } else if (ch == '-'){
                // decrease training rate
                auto oldRate = rate.load(std::memory_order_relaxed);
                const auto newRate = oldRate / 1.1;
                std::cout << "The learning rate is now " << newRate << '\n';
                rate.store(newRate, std::memory_order_relaxed);
            } else if (ch == ']'){
                // increase momentum
                const auto oldMomentum = momentum.load(std::memory_order_relaxed);
                const auto newMomentum = std::min(oldMomentum + 0.05, 1.0);
                std::cout << "The momentum ratio is now " << newMomentum << '\n';
                momentum.store(newMomentum, std::memory_order_relaxed);
            } else if (ch == '['){
                // decrease momentum
                auto oldMomentum = momentum.load(std::memory_order_relaxed);
                const auto newMomentum = std::max(oldMomentum - 0.05, 0.0);
                std::cout << "The momentum ratio is now " << newMomentum << '\n';
                momentum.store(newMomentum, std::memory_order_relaxed);
            } else if (ch == 'i'){
                std::cout << "Info:\n";
                std::cout << "    The learning rate is " << rate.load(std::memory_order_relaxed) << '\n';
                std::cout << "    The momentum ratio is " << momentum.load(std::memory_order_relaxed) << '\n';
            } else if (ch == 'e'){
                std::cout << "Do you want to end training? (y/n)";
                std::cin >> ch;
                if (ch == 'y'){
                    done = true;
                }
            } else {
                std::cout << "\nWhat??\n";
            }
            networkMutex.unlock();
        }
    }};

    trainUntil(0.005);

    done = true;
    inputThread.join();
}

int main(){
    std::cout << std::fixed;
    std::cout.precision(15);

    const auto train = [](){
        sf::SoundBuffer trainingBuff;
        std::cout << "Loading training sound...";
        trainingBuff.loadFromFile("../sound/train.wav");
        std::cout << " done.\n";

        prepareTrainingData(trainingBuff);

        network.randomizeWeights();

        trainNetwork(trainingBuff);
    };

    const auto test = [](){
        const std::size_t testBegin = 0;
        const std::size_t testEnd = (108 - 30) * 4;

        sf::SoundBuffer testBuff;
        std::cout << "Loading test sound...";
        testBuff.loadFromFile("../sound/test.wav");
        std::cout << " done.\n";

        for (std::size_t i = testBegin; i < testEnd; i++){
            const auto [label, conf] = predict(testBuff, i);
            std::cout << i << " - " << noteNames[label] << '(' << label << "), confidence: " << conf << '\n';
        }
    };

    const auto restore = [](){
        std::cout << "Please enter a file name:\n";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (std::string line; std::getline(std::cin, line)){
            try {
                network.loadWeights(line);
                std::cout << "Done.\n";
            } catch (...) {
                std::cout << "That's a bad file, sorry.\n";
                return false;
            }
        } else {
            std::cout << "Dang, that didn't work.\n";
            return false;
        }
        return true;
    };

    std::cout << "Welcome to flutelearner! Please choose an option:\n";
    std::cout << "    t - load the training data and train the network\n";
    std::cout << "    r - restore the network from a file and test it\n";
    char ch;
    std::cin >> ch;
    if (ch == 't'){
        train();
        test();
    } else if (ch == 'r'){
        if (!restore()){
            return 2;
        }
        test();
    } else {
        std::cout << "\nwhat?\n\n";
        return 1;
    }

    return 0;
}