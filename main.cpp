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
// shouldn't really be needed (the highest expected note frequency
// is ~988 Hz)
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

// Labelled input-output pairs of training data.
// The first element in each pair is the note's index (its start time
// multiplied by 4 notes per second), and the second element in each
// pair is the expected note number at that time (0 denotes silence;
// see the note names above)
const auto trainingLabels = std::vector<std::pair<std::size_t, std::size_t>>{
    // First batch of training data: one note every quarter second
    {0,  1 }, {1,  2 }, {2,  3 }, {3,  4 }, {4,  5 }, {5,  6 },
    {6,  7 }, {7,  8 }, {8,  9 }, {9,  10}, {10, 11}, {11, 12},
    {12, 13}, {13, 14}, {14, 15}, {15, 16}, {16, 17}, {17, 18},
    {18, 19}, {19, 20}, {20, 21}, {21, 22}, {22, 23}, {23, 24},
    {24, 25}, {25, 26}, {26, 27}, {27, 28}, {28, 29}, {29, 30},
    {30, 31}, {31, 32}, {32, 33}, {33, 34}, {34, 35}, {35, 36},
        
    // learn some silence too
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
// magnitudes of (some of) the positive frequency components
// buff             : the input sound
// noteIndex        : the note's starting time, multiplied by 4 notes per second
// samplesOffset    : A number of samples by which the entire window is shifted
//                    while reading from the input sound.
//                    This is used to add some variation to the training data
// noiseAmplitude   : The amplitude of white noise which will be added to the
//                    input sound before taking the Fourier transform. This is
//                    also used to add some variation to the training data
//                    and to encourage robustness.
std::vector<double> getNoteFrequences(const sf::SoundBuffer& buff, std::size_t noteIndex, std::size_t samplesOffset, double noiseAmplitude){

    // the starting sample at which to read the input sound
    const std::size_t startSample = noteIndex * noteLength + samplesOffset;

    // assert that the audio is mono. Stereo audio uses interleaved samples
    // and I don't want to deal with that.
    if (buff.getChannelCount() != 1){
        throw std::runtime_error("I though the input was supposed to be mono audio :(");
    }
    const auto samples = buff.getSamples();

    // A vector of complex numbers for taking the FFT
    std::vector<std::complex<double>> x;
    x.reserve(noteLengthTruncated);

    // conversion factor from 16-bit integer audio to normalized floating point
    const auto k = 1.0 / static_cast<double>(std::numeric_limits<sf::Int16>::max());

    // uniform random distribution used for adding white noise
    const auto dist = std::uniform_real_distribution<double>{-1.0, 1.0};

    // read entire window from input sound
    for (std::size_t i = 0; i < noteLengthTruncated; ++i){
        // get the i'th sample
        const auto s = samples[startSample + i];
        // compute the Hann window function
        const auto win = 0.5 * (1.0 - std::cos(static_cast<double>(i) * pi / static_cast<double>(noteLengthTruncated)));
        // convert to floating point, apply window, add noise, and add to complex vector
        x.push_back((static_cast<double>(s) * k + noiseAmplitude * dist(randEng)) * win);
    }

    // take the Fourier transform
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

// The neural network structure being used, with:
// - numFrequencies (512) output neurons
// - a layer of 256 hidden neurons
// - a layer of 128 hidden neurons
// - a layer of 64 hidden neurons
// - numNotes (37) output neurons
// The presence of a note will be indicated by a 1 at that note's
// corresponding output neuron
using NetworkType = NeuralNetwork<numNotes, 64, 128, 256, numFrequences>;

// the neural network object
auto network = NetworkType{};

// given a sound and note index, predict the note that is being played
// Returns a pair containing, in order:
// - the estimated note (see noteNames for interpretation)
// - the confidence of that estimate, between 0 and 1.
std::pair<std::size_t, double> predict(const sf::SoundBuffer& buff, std::size_t noteIndex){
    // get note frequencies at the note index (with no offset or white noise)
    auto x = getNoteFrequences(buff, noteIndex, 0, 0.0);
    // compute the network's prediction
    auto y = network.compute(x);
    // find the output neuron with the highest activation
    auto maxIt = std::max_element(y.begin(), y.end());
    assert(maxIt != y.end());
    // return that neuron's index and its value
    return {maxIt - y.begin(), *maxIt};
}

// Computes how well the network predicts all the training examples
// Returns the fraction of training examples which are correctly
// labeled, using predict()
double trainingAccuracy(const sf::SoundBuffer& buff){
    std::size_t numCorrect = 0;
    for (const auto& inputAndLabel : trainingLabels){
        const auto [actualLabel, conf] = predict(buff, inputAndLabel.first);
        if (inputAndLabel.second == actualLabel){
            ++numCorrect;
        }
    }
    return static_cast<double>(numCorrect) / static_cast<double>(trainingLabels.size());
}

// physical storage for training data data
std::vector<std::pair<std::vector<double>, std::vector<double>>> trainingData;

// Given the training sound, prepares the training data from
// the labelled set of training example (see trainingLabels)
void prepareTrainingData(const sf::SoundBuffer& buff){
    std::cout << "Preparing training data...\n";

    // number of times each traning note is repeated
    // (with random noise and time offset)
    const std::size_t upsampling = 128;

    // For printing a progress bar
    int lastWidth = 0;
    int targetWidth = 60;
    int count = 0;
    int total = trainingLabels.size() * upsampling;
    std::cout << "0%";
    for (int i = 0; i < targetWidth; ++i){
        std::cout << ' ';
    }
    std::cout << "100%\n";
    const auto updateProgress = [&](){
        ++count;
        auto width = targetWidth * count / total;
        while (width > lastWidth){
            std::cout << '|';
            ++lastWidth;
        }
    };

    // maximum amplitude of random amount of white noise that is added to signal
    const double maxNoiseAmp = 0.1;
    auto noiseDist = std::uniform_real_distribution<double>{0.0, maxNoiseAmp};

    // maximum sample offset by which each training example is randomly shifted
    const std::size_t maxOffset = noteLengthTruncated / 8;
    auto offsetDist = std::uniform_int_distribution<std::size_t>{0, maxOffset};

    // for every labelled training example
    for (const auto& noteIndexAndLabel : trainingLabels){
        // repeat the training examples a number of times
        for (std::size_t i = 0; i < upsampling; ++i){
            // get the frequencies of the note for that training example,
            // after adding a random time offset and random amount of white noise
            auto input = getNoteFrequences(buff, noteIndexAndLabel.first, offsetDist(randEng), noiseDist(randEng));

            // the output must be all zeroes, except for the expect note
            // which must be a one.
            auto output = std::vector<double>(numNotes, 0.0);
            output.at(noteIndexAndLabel.second) = 1.0;

            // add to the vector of training data
            trainingData.push_back(std::pair{std::move(input), std::move(output)});

            // give some visual feedback
            updateProgress();
        }
    }
    std::cout << " done.\n";
}

// Returns a view to a randomized subset of the training data,
// for use in stochastic gradient descent
std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> randomTrainingBatch(std::size_t size){
    std::vector<std::pair<NetworkType::InputType, NetworkType::OutputType>> ret;

    auto dist = std::uniform_int_distribution<std::size_t>{0, trainingData.size() - 1};

    for (std::size_t i = 0; i < size; ++i){
        const auto& [input, output] = trainingData.at(dist(randEng));
        ret.push_back(std::pair{NetworkType::InputType{input}, NetworkType::OutputType{output}});
    }
    
    return ret;
}

// train the neural network
void trainNetwork(const sf::SoundBuffer& buff){
    // NOTE: atomics and mutexes are being used here
    // because the input is read from a separate thread
    // (so that reading from the standard input doesn't
    // block training)

    // the learning rate
    std::atomic<double> rate = 0.05;

    // the momentum ratio (fraction of previous
    // gradient that is added to current step)
    std::atomic<double> momentum = 0.9;

    // mutexes used to allow input thread to block training thread
    // right away. A single mutex would allow the training thread
    // to run any number of times before the input thread succeeds.
    // https://stackoverflow.com/questions/11666610/how-to-give-priority-to-privileged-thread-in-mutex-locking
    std::mutex networkMutex, nextUpMutex, lowPriorityMutex;

    // flag used to end training and input threads
    std::atomic<bool> done = false;

    // helper function to train the network in batches
    // until a desired loss is reached
    const auto trainUntil = [&](double desiredLoss){
        // the total number of training iterations
        std::size_t count = 0;
        // the current loss
        auto loss = 1e6;

        while (loss > desiredLoss && !done){
            // in sets of 50...
            for (size_t i = 0; i < 50 && !done.load(std::memory_order_relaxed); ++i){
                // get control
                lowPriorityMutex.lock();
                nextUpMutex.lock();
                networkMutex.lock();
                nextUpMutex.unlock();

                // zero the loss (for accumulating an average)
                loss = 0.0;
                // in batches of 256...
                const auto batchSize = 1 << 8;
                for (size_t i = 0; i < batchSize; ++i){
                    // get some randomized training data
                    auto d = randomTrainingBatch(8);
                    // backpropagate and take a step in the right direction
                    loss += network.takeStep(
                        d,
                        rate.load(std::memory_order_relaxed),
                        momentum.load(std::memory_order_relaxed)
                    );
                    ++count;
                }
                // compute and display the average loss during this batch
                loss /= static_cast<double>(batchSize);
                std::cout << "loss = " << loss << '\n';

                // release control
                networkMutex.unlock();
                lowPriorityMutex.unlock();
            }
            std::cout << "    training accuracy = " << trainingAccuracy(buff) * 100.0 << "%\n";
            std::cout << "    total iterations so far: " << count << '\n';
        }
    };

    // Start a thread for reading input whose entry point is as follows
    auto inputThread = std::thread{[&](){
        char ch = {};
        // while there is work to do, get the next character of input
        while (!done.load(std::memory_order_relaxed) && std::cin >> ch){
            // get control
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
                // print training information
                std::cout << "Info:\n";
                std::cout << "    The learning rate is " << rate.load(std::memory_order_relaxed) << '\n';
                std::cout << "    The momentum ratio is " << momentum.load(std::memory_order_relaxed) << '\n';
            } else if (ch == 'e'){
                // end the training process
                std::cout << "Do you want to end training? (y/n)";
                std::cin >> ch;
                if (ch == 'y'){
                    done = true;
                }
            } else {
                // help a confused user
                std::cout << "\nWhat??\n";
                std::cout << "These are the actions you can perform. To perform an action\n";
                std::cout << "while the network is training, type that action's character\n";
                std::cout << "and press Enter.\n\n";
                std::cout << "    =    Increase the learning rate\n";
                std::cout << "    -    Decrease the learning rate\n";
                std::cout << "    ]    Increase the momentum ratio\n";
                std::cout << "    [    Decrease the momentum ratio\n";
                std::cout << "    s    Save the network's state to a file\n";
                std::cout << "    r    Restore the network's state from a file\n";
                std::cout << "    e    End training\n";
                std::cout << "    i    See the current learning rate and momentum ratio\n";
                std::cout << '\n';
                std::cout << "         Enter any other character to display this help.\n";
                std::cout << "\nEnter any key to continue training...";
                std::cin >> ch;
                std::cout << '\n';
            }
            // release control
            networkMutex.unlock();
        }
    }};

    // Train until some small amount of loss
    trainUntil(0.01);

    // kill the input thread
    done = true;
    inputThread.join();
}

int main(){
    // Lots of floating point precision to make training
    // seem like it isn't plateauing out
    std::cout << std::fixed;
    std::cout.precision(15);

    // Helper function to load training data and train network
    const auto train = [](){
        sf::SoundBuffer trainingBuff;
        std::cout << "Loading training sound...";
        trainingBuff.loadFromFile("../sound/train.wav");
        std::cout << " done.\n";

        prepareTrainingData(trainingBuff);

        network.randomizeWeights();

        trainNetwork(trainingBuff);
    };

    // helper function to load test sound and predict its notes
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

    // helper function to restore the network's state from a file
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
        std::cout << "\nWhat?\n\n";
        return 1;
    }

    return 0;
}