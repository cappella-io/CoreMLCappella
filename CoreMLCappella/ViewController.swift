//
//  ViewController.swift
//  CoreMLCappella
//
//  Created by Arshad Awati on 06/06/24.
//

import UIKit
import AVFoundation
import Accelerate
import CoreML

class ViewController: UIViewController {

    var melSpectrogramModelURL: URL?
    var babyCryDetectionModelURL: URL?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        // Load the models
        //        guard let melSpectrogramModelURL = Bundle.main.url(forResource: "Mel_spectogram", withExtension: "mlpackage"),
        //              let babyCryDetectionModelURL = Bundle.main.url(forResource: "encodedBabyCryDetectionModel", withExtension: "mlpackage") else {
        //            fatalError("Model files not found.")
        //        }
        //        print("Loading Mel Spectrogram model...")
        //        let melSpectrogramModel = try! MLModel(contentsOf: melSpectrogramModelURL)
        //        print("Mel Spectrogram model loaded and set to evaluation mode")
        //        print("Loading Baby Cry Detection model...")
        //        let babyCryDetectionModel = try! MLModel(contentsOf: babyCryDetectionModelURL)
        //        print("Baby Cry Detection model loaded and set to evaluation mode")
        
        if let url = Bundle.main.url(forResource: "35", withExtension: "mp3") {
            print("URL:", url)
            if let result = detectBabyCry(from: url) {
                print("Prediction: \(result ? "Baby Cry Detected" : "No Baby Cry Detected")")
            } else {
                print("Failed to detect baby cry.")
            }
        }
    }
    
    func loadAudioFileAsMono(url: URL) -> [Float]? {
        do {
            // Load the audio file
            let file = try AVAudioFile(forReading: url)
            
            // Get the format
            let format = file.processingFormat
            
            // Check the number of channels
            let channels = format.channelCount
            
            let frameCount = AVAudioFrameCount(file.length)
            
           
            // Create a buffer to hold the audio data
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)) else {
                return nil
            }
            
            let numberOfFrames = Int(buffer.frameLength)
            // Print the shape of the channels
            print("Shape of channels: (\(channels), \(numberOfFrames))")
            
            // Read the audio data into the buffer
            try file.read(into: buffer)
            
            // If the audio is already mono, return the samples
            if channels == 1 {
                return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
            }
            
            // Otherwise, average the channels to create a mono signal
            let leftChannel = buffer.floatChannelData![0]
            let rightChannel = buffer.floatChannelData![1]
            var monoSignal = [Float](repeating: 0, count: Int(buffer.frameLength))
            
            for i in 0..<5/*Int(buffer.frameLength)*/ {
                print("Left Channel[\(i)]:", leftChannel[i])
                print("Right Channel[\(i)]:", rightChannel[i])
                monoSignal[i] = (leftChannel[i] + rightChannel[i]) / 2.0
                print("MonoSignal[\(i)]:", monoSignal[i])
            }
            
            print("Count of left channel: \(leftChannel)")
            
            
            let leftChannelLength = Int(buffer.frameLength)
            print("Length of left channel: \(leftChannelLength)")
            
            return monoSignal
        } catch {
            print("Error loading audio file: \(error.localizedDescription)")
            return nil
        }
    }
        //working core ml ---start
        func loadAudioFile(url: URL) -> AVAudioPCMBuffer? {
            let audioFile = try? AVAudioFile(forReading: url)
            guard let file = audioFile else { return nil }
            
//            let format = file.processingFormat
            let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: audioFile!.fileFormat.sampleRate, channels: AVAudioChannelCount(audioFile!.fileFormat.channelCount), interleaved: false)
            print("format:", format!)
            let frameCount = UInt32(file.length)
            print("file length:", frameCount)
            let buffer = AVAudioPCMBuffer(pcmFormat: format!, frameCapacity: frameCount)
            print("buffer:", buffer!)
            let sampleRate = file.fileFormat.sampleRate
            print("sampleRate:", sampleRate)
            
            // Convert AVAudioPCMBuffer to Data
//            let dataSize = Int(buffer!.frameLength) * format!.streamDescription.pointee.mBytesPerFrame
//                    let data = Data(bytes: buffer.floatChannelData!.pointee, count: dataSize)

            var audioFormat = audioFile!.processingFormat
            var samples = [Float]()
            let channels = 1//Int(audioFormat.channelCount)
//            let sampleRate =
            do {
                try audioFile!.read(into: buffer!)
                } catch {
                    print("Error reading buffer: \(error)")
                    return nil
                }
                
            let data = buffer?.floatChannelData!
                
                for frame in 0..<Int(frameCount) {
                    for channel in 0..<channels {
                        samples.append(data![channel][frame])
                    }
                }
            print("Audio Count:", samples.count)
            print("Audio Array:", samples[0...2])
            
            try? file.read(into: buffer!)
            
            return buffer
        }
        
        func audioBufferToTensor(buffer: AVAudioPCMBuffer) -> MLMultiArray? {
            
            let frameLengths = Int(buffer.frameLength)
            let audioTensor = try! MLMultiArray(shape: [1, frameLengths as NSNumber], dataType: .float32)
            let audioPointer = buffer.floatChannelData![0]
            print("Original audio data (first 5 samples):")
            for i in 0..<5 {
                print("y[\(i)]: \(audioPointer[i])")
            }
            print("Original audio data shape: (\(frameLengths))")
            
            
            let frameLength = Int(buffer.frameLength)
            print("frameLength:", frameLength)
            let channels = Int(buffer.format.channelCount)
            print("frameLength:", frameLength)
            let sample = Int(buffer.format.sampleRate)
            
            let shape = [1, 44100] as [NSNumber] //[1, frameLength, channels] as [NSNumber]
            
            guard let tensor = try? MLMultiArray(shape: shape, dataType: .float32) else {
                return nil
            }
            
            let channelData = buffer.floatChannelData![0]
            print("channelData:", channelData)
            for i in 0..<frameLength {
                tensor[[0, NSNumber(value: i), 0]] = NSNumber(value: channelData[i])
            }
            
//            var audioData: [Float] = []
//                   for channel in 0..<channels {
//                       let channelDataBuffer = channelData[channel]
//                       let channelDataArray = UnsafeBufferPointer<Any>(start: channelDataBuffer, count: samples)
//                       audioData.append(contentsOf: channelDataArray)
//                   }
//            
            print("Audio Tensor:", tensor)
            return tensor
        }
        
        func predictBabyCry(melSpectrogram: MLMultiArray) -> Bool? {
            do {
                let model = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
                let input = encodedBabyCryDetectionModelInput(audio: melSpectrogram)
                print("Input:", input)
                let prediction = try model.prediction(input: input)
                
                // Assuming the output property is a float value indicating the logit
                let logitArray = prediction.var_104ShapedArray // Replace 'output' with the actual property name
                print("Logit array:", logitArray)
                // Extract the logit value from the MLShapedArray
                let logitSlice = logitArray[0]
                guard let logit = logitSlice.scalar else {
                    print("Failed to extract logit value.")
                    return nil
                }
                print("Logit:", logit)
                let probability = 1 / (1 + exp(-logit))
                print("probability:", probability)
                // Return the binary prediction based on a threshold of 0.5
                return probability > 0.5
            } catch {
                print("Failed to make prediction: \(error)")
                return nil
            }
        }
        
        // Function to process audio and get a mel spectrogram
//        func processAudio(_ audioData: AVAudioPCMBuffer) -> MLMultiArray? {
//            
//            let frameLength = Int(audioData.frameLength)
//            let audioTensor = try! MLMultiArray(shape: [1, frameLength as NSNumber], dataType: .float32)
//            let audioPointer = audioData.floatChannelData![0]
//            print("Original audio data (first 5 samples):")
//            for i in 0..<5 {
//                print("y[\(i)]: \(audioPointer[i])")
//            }
//            print("Original audio data shape: (\(frameLength))")
//            let maxVal = vDSP.maximum(audioPointer, n: frameLength)
//            print("Maximum value in original audio data: \(maxVal)")
//            let normalizedPointer = UnsafeMutablePointer<Float>.allocate(capacity: frameLength)
//            defer { normalizedPointer.deallocate() }
//            vDSP_vsdiv(audioPointer, 1, [maxVal], normalizedPointer, 1, vDSP_Length(frameLength))
//            print("Normalized audio data (first 5 samples):")
//            for i in 0..<5 {
//                print("y[\(i)]: \(normalizedPointer[i])")
//            }
//            print("Normalized audio data shape: (\(frameLength))")
//            for i in 0..<frameLength {
//                audioTensor[i] = NSNumber(value: normalizedPointer[i])
//            }
//            let melSpectrogramInput = try! MLDictionaryFeatureProvider(dictionary: ["input": audioTensor])
//            print("Generating mel spectrogram...")
//            let melSpectrogramOutput = try! melSpectrogramModel.prediction(from: melSpectrogramInput)
//            let melSpectrogram = melSpectrogramOutput.featureValue(for: "output")?.multiArrayValue
//            if let melSpectrogram = melSpectrogram {
//                let melShape = melSpectrogram.shape
//                print("Mel Spectrogram shape: (\(melShape[1]), \(melShape[2]))")
//                print("Mel Spectrogram (first 5 samples):")
//                for i in 0..<5 {
//                    for j in 0..<5 {
//                        print("\(melSpectrogram[i * melShape[2].intValue + j]) ", terminator: "")
//                    }
//                    print()
//                }
//                // Assuming melSpectrogram is in dB already, otherwise convert it
//                print("Mel Spectrogram (dB, first 5 samples):")
//                for i in 0..<5 {
//                    for j in 0..<5 {
//                        let value = melSpectrogram[i * melShape[2].intValue + j]
//                        print("\(value) ", terminator: "")
//                    }
//                    print()
//                }
//            }
//            return melSpectrogram
//        }
        
        // Function to detect baby cry
//        func detectBabyCry(spectrogram: MLMultiArray) -> (Float, Float, Int) {
//            let babyCryInput = try! MLDictionaryFeatureProvider(dictionary: ["input": spectrogram])
//            print("Running Baby Cry Detection model...")
//            let babyCryOutput = try! babyCryDetectionModel.prediction(from: babyCryInput)
//            let rawOutput = babyCryOutput.featureValue(for: "output")!.floatValue
//            let sigmoidOutput = 1 / (1 + exp(-rawOutput))
//            let roundedPrediction = Int(round(sigmoidOutput))
//            return (rawOutput, sigmoidOutput, roundedPrediction)
//        }
        
        func detectBabyCry(from url: URL) -> Bool? {
            let monoFloat = self.loadAudioFileAsMono(url: url)
            print("Mono Float:", monoFloat!)
            guard let audioBuffer = loadAudioFile(url: url),
                  let audioTensor = audioBufferToTensor(buffer: audioBuffer) else {
                return nil
            }
            
            print("Audio Tesnsor:", audioTensor)
            
            let melSpectrogramGenerator = MelSpectrogram()
            guard let melSpectrogram = melSpectrogramGenerator.generate(from: audioTensor) else {
                return nil
            }
            
            print("Melspectrogram:", melSpectrogram)
            return predictBabyCry(melSpectrogram: melSpectrogram)
        }
    
    //working core ml ---end


}

