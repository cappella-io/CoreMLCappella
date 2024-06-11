//
//  MelSpectrogramGenerator.swift
//  CoreMLCappella
//
//  Created by Arshad Awati on 07/06/24.
//

//let sampleRate: Float = 16000.0
//let fftSize: Int = 400
//let hopLength: Int = 160
//let numMels: Int = 80

import Foundation
import CoreML
import Accelerate

//class MelSpectrogram {
//    func generate(from tensor: MLMultiArray) -> MLMultiArray? {
//        // Ensure the tensor shape and type are correct
//        guard tensor.dataType == .float32 else {
//            return nil
//        }
//        
//        // Extract audio signal from the tensor
//        let audioSignal = Array(UnsafeBufferPointer(start: tensor.dataPointer.assumingMemoryBound(to: Float.self), count: tensor.count))
//        
//        // Define parameters for STFT and Mel Spectrogram
//        let sampleRate: Float = 16000.0
//        let fftSize: Int = 400
//        let hopLength: Int = 160
//        let numMels: Int = 80
//        
//        // Perform Short-Time Fourier Transform (STFT)
//        let stft = computeSTFT(audioSignal: audioSignal, fftSize: fftSize, hopLength: hopLength)
//        
//        // Apply Mel filter bank
//        let melFilterBank = createMelFilterBank(sampleRate: sampleRate, fftSize: fftSize, numMels: numMels)
//        let melSpectrogram = applyMelFilterBank(stft: stft, melFilterBank: melFilterBank)
//        
//        // Convert to MLMultiArray
//        let melSpectrogramShape: [NSNumber] = [1, 1, NSNumber(value: numMels), NSNumber(value: melSpectrogram[0].count)]
//        guard let melSpectrogramArray = try? MLMultiArray(shape: melSpectrogramShape, dataType: .float32) else {
//            return nil
//        }
//        
//        // Populate the MLMultiArray with the Mel spectrogram data
//        for i in 0..<numMels {
//            for j in 0..<melSpectrogram[0].count {
//                melSpectrogramArray[[0, 0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: melSpectrogram[i][j])
//            }
//        }
//        
//        return melSpectrogramArray
//    }
//    
//    private func computeSTFT(audioSignal: [Float], fftSize: Int, hopLength: Int) -> [[Float]] {
//        let frameCount = (audioSignal.count - fftSize) / hopLength + 1
//        var stft: [[Float]] = []
//        
//        var window = [Float](repeating: 0, count: fftSize)
//        vDSP_hann_window(&window, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
//        
//        var real = [Float](repeating: 0, count: fftSize / 2)
//        var imag = [Float](repeating: 0, count: fftSize / 2)
//        var splitComplex = DSPSplitComplex(realp: &real, imagp: &imag)
//        
//        for i in 0..<frameCount {
//            let start = i * hopLength
//            let end = start + fftSize
//            let frame = Array(audioSignal[start..<end])
//            
//            var windowedFrame = [Float](repeating: 0, count: fftSize)
//            vDSP_vmul(frame, 1, window, 1, &windowedFrame, 1, vDSP_Length(fftSize))
//            
//            windowedFrame.withUnsafeBufferPointer { pointer in
//                pointer.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftSize) {
//                    vDSP_ctoz($0, 2, &splitComplex, 1, vDSP_Length(fftSize / 2))
//                }
//            }
//            
//            vDSP_fft_zip(vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), vDSP_DFT_Direction.FORWARD)!, &splitComplex, 1, vDSP_Length(log2(Float(fftSize))), vDSP_DFT_Direction.FORWARD.rawValue)
//            
//            var magnitude = [Float](repeating: 0, count: fftSize / 2)
//            vDSP_zvmags(&splitComplex, 1, &magnitude, 1, vDSP_Length(fftSize / 2))
//            vvsqrtf(&magnitude, magnitude, [Int32(fftSize / 2)])
//            
//            stft.append(magnitude)
//        }
//        
//        return stft
//    }
//    
//    private func createMelFilterBank(sampleRate: Float, fftSize: Int, numMels: Int) -> [[Float]] {
//            let nyquist = sampleRate / 2.0
//            let melMin = 0.0
//            let melMax = 2595.0 * log10(1.0 + nyquist / 700.0)
//            
//            var melPoints = [Float]()
//            let op1 = (melMax - Float(melMin)) / Float(numMels + 1)
//            for i in 0..<(numMels + 2) {
//                melPoints.append(Float(melMin) + Float(i) * op1)
//            }
//            
//            let hzPoints = melPoints.map { 700.0 * (pow(10.0, $0 / 2595.0) - 1.0) }
//            let binPoints = hzPoints.map { Int($0 / nyquist * Float(fftSize / 2)) }
//            
//            var filterBank: [[Float]] = Array(repeating: [Float](repeating: 0, count: fftSize / 2), count: numMels)
//            
//            for i in 1..<binPoints.count - 1 {
//                let left = binPoints[i - 1]
//                let center = binPoints[i]
//                let right = binPoints[i + 1]
//                
//                for j in left..<center {
//                    filterBank[i - 1][j] = (Float(j) - Float(left)) / (Float(center) - Float(left))
//                }
//                
//                for j in center..<right {
//                    filterBank[i - 1][j] = 1.0 - (Float(j) - Float(center)) / (Float(right) - Float(center))
//                }
//            }
//            
//            return filterBank
//        }
//    
//    private func applyMelFilterBank(stft: [[Float]], melFilterBank: [[Float]]) -> [[Float]] {
//        var melSpectrogram: [[Float]] = []
//        
//        for i in 0..<melFilterBank.count {
//            var melSpectrum = [Float](repeating: 0, count: stft[0].count)
//            
//            for j in 0..<stft[0].count {
//                for k in 0..<stft.count {
//                    melSpectrum[j] += stft[k][j] * melFilterBank[i][k]
//                }
//            }
//            
//            melSpectrogram.append(melSpectrum)
//        }
//        
//        return melSpectrogram
//    }
//}

class MelSpectrogram {
    func generate(from tensor: MLMultiArray) -> MLMultiArray? {
        // Placeholder for your custom Mel spectrogram generation logic.
                // This should convert the audio tensor to a Mel spectrogram tensor.
                
                // Example shape [1, 1, 40, 173]
                let melSpectrogramShape: [NSNumber] = [1, 1, 40, 173]
                
                guard let melSpectrogram = try? MLMultiArray(shape: melSpectrogramShape, dataType: .float32) else {
                    return nil
                }
                
                // Fill melSpectrogram with your data.
                // For demonstration purposes, we'll fill it with dummy data.
                // Replace this with actual Mel spectrogram calculation.
                let numMels = 40
                let numFrames = 173
                
                for i in 0..<numMels {
                    for j in 0..<numFrames {
                        let value: Float = Float.random(in: 0..<1)  // Replace with actual spectrogram value
                        melSpectrogram[[0, 0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: value)
                    }
                }
                return melSpectrogram
            }
}
