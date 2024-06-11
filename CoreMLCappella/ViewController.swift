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
        
        //working core ml ---start
        func loadAudioFile(url: URL) -> AVAudioPCMBuffer? {
            let audioFile = try? AVAudioFile(forReading: url)
            guard let file = audioFile else { return nil }
            
            let format = file.processingFormat
            let frameCount = UInt32(file.length)
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount)
            try? file.read(into: buffer!)
            
            return buffer
        }
        
        func audioBufferToTensor(buffer: AVAudioPCMBuffer) -> MLMultiArray? {
            let frameLength = Int(buffer.frameLength)
            print("frameLength:", frameLength)
            let channels = Int(buffer.format.channelCount)
            print("frameLength:", frameLength)

            let shape = [1, 44100] as [NSNumber] //[1, frameLength, channels] as [NSNumber]
            
            guard let tensor = try? MLMultiArray(shape: shape, dataType: .float32) else {
                return nil
            }
            
            let channelData = buffer.floatChannelData![0]
            for i in 0..<frameLength {
                tensor[[0, NSNumber(value: i), 0]] = NSNumber(value: channelData[i])
            }
            
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
        
        func detectBabyCry(from url: URL) -> Bool? {
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
    }
    //working core ml ---end


}

