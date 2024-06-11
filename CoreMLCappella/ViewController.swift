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

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        if let url = Bundle.main.url(forResource: "35", withExtension: "mp3") {
            print("URL:", url)
            if let result = detectBabyCry(from: url) {
                print("Prediction: \(result ? "Baby Cry Detected" : "No Baby Cry Detected")")
            } else {
                print("Failed to detect baby cry.")
            }
            
        }
        
        // working core ml ---start
        func loadAudioFile(url: URL) -> AVAudioPCMBuffer? {
            print("Loading audio file from URL:", url)
            let audioFile = try? AVAudioFile(forReading: url)
            guard let file = audioFile else {
                print("Failed to load audio file.")
                return nil
            }
            
            let format = file.processingFormat
            print("Processing format:", format)
            let frameCount = UInt32(file.length)
            print("Frame count:", frameCount)
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            if buffer == nil {
                print("Failed to create AVAudioPCMBuffer.")
                return nil
            }
            
            print("Buffer created with frame capacity:", frameCount)
            let sampleRate = file.fileFormat.sampleRate
            print("Sample rate:", sampleRate)

            do {
                try file.read(into: buffer!)
                print("Buffer read successfully.")
            } catch {
                print("Error reading buffer: \(error)")
                return nil
            }
                
            guard let data = buffer?.floatChannelData else {
                print("Failed to get float channel data from buffer.")
                return nil
            }
                
            var samples = [Float]()
            let channels = Int(buffer!.format.channelCount)
            print("Channels:", channels)
            for frame in 0..<Int(frameCount) {
                for channel in 0..<channels {
                    samples.append(data[channel][frame])
                }
            }
            print("Audio sample count:", samples.count)
            print("First few audio samples:", samples.prefix(3))
            
            return buffer
        }
        
        func audioBufferToTensor(buffer: AVAudioPCMBuffer) -> MLMultiArray? {
            print("Converting audio buffer to tensor.")
            let frameLength = Int(buffer.frameLength)
            print("Frame length:", frameLength)
            let channels = Int(buffer.format.channelCount)
            print("Channels:", channels)
            let sampleRate = Int(buffer.format.sampleRate)
            print("Sample rate:", sampleRate)
            
            let shape = [1, frameLength] as [NSNumber] // Adjusted shape to match frameLength
            guard let tensor = try? MLMultiArray(shape: shape, dataType: .float32) else {
                print("Failed to create MLMultiArray.")
                return nil
            }
            
            let channelData = buffer.floatChannelData![0]
            print("Channel data pointer:", channelData)
            for i in 0..<frameLength {
                tensor[i] = NSNumber(value: channelData[i])
            }
            
            print("Audio tensor created:", tensor)
            return tensor
        }
        
        func predictBabyCry(melSpectrogram: MLMultiArray) -> Bool? {
            print("Predicting baby cry from mel spectrogram.")
            do {
                let model = try encodedBabyCryDetectionModel(configuration: MLModelConfiguration())
                let input = encodedBabyCryDetectionModelInput(audio: melSpectrogram)
                print("Model input created:", input)
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
                print("Logit value:", logit)
                let probability = 1 / (1 + exp(-logit))
                print("Probability:", probability)
                // Return the binary prediction based on a threshold of 0.5
                return probability > 0.5
            } catch {
                print("Failed to make prediction: \(error)")
                return nil
            }
        }
        
        
        func detectBabyCry(from url: URL) -> Bool? {
            print("Detecting baby cry from URL:", url)
            guard let audioBuffer = loadAudioFile(url: url) else {
                print("Failed to load audio file.")
                return nil
            }
            
            guard let audioTensor = audioBufferToTensor(buffer: audioBuffer) else {
                print("Failed to convert audio buffer to tensor.")
                return nil
            }
            
            print("Audio tensor:", audioTensor)
            
            let melSpectrogramGenerator = MelSpectrogram()
            guard let melSpectrogram = melSpectrogramGenerator.generate(from: audioTensor) else {
                print("Failed to generate mel spectrogram.")
                return nil
            }
            
            print("Mel spectrogram:", melSpectrogram)
            return predictBabyCry(melSpectrogram: melSpectrogram)
        }
    }
    // working core ml ---end

}


