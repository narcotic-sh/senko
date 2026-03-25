/*
    Credit: https://github.com/FluidInference/FluidAudio
    Some parts (like ANE memory stuff) copied verbatim.
    Other parts adapted / lightly modified.
*/

import Foundation
import CoreML
import Darwin
import Accelerate

@available(macOS 13.0, iOS 16.0, *)
public enum ANEMemoryUtils {
    /// ANE requires 64-byte alignment for optimal DMA transfers
    public static let aneAlignment = 64

    /// ANE tile size for matrix operations
    public static let aneTileSize = 16

    public enum ANEMemoryError: Error {
        case allocationFailed
        case invalidShape
        case unsupportedDataType
    }

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public static func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType,
        zeroClear: Bool = true
    ) throws -> MLMultiArray {
        // Calculate element size
        let elementSize = getElementSize(for: dataType)

        // Calculate optimal strides for ANE
        let strides = calculateOptimalStrides(for: shape)

        // Calculate actual elements needed based on strides (accounts for padding)
        let totalElementsNeeded: Int
        if !shape.isEmpty {
            totalElementsNeeded = strides[0].intValue * shape[0].intValue
        } else {
            totalElementsNeeded = 0
        }

        // Align the allocation size to ANE requirements
        let bytesNeeded = totalElementsNeeded * elementSize
        let alignedBytes = max(aneAlignment, ((bytesNeeded + aneAlignment - 1) / aneAlignment) * aneAlignment)

        // Allocate page-aligned memory for ANE DMA
        var alignedPointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&alignedPointer, aneAlignment, alignedBytes)

        guard result == 0, let pointer = alignedPointer else {
            throw ANEMemoryError.allocationFailed
        }

        // Zero-initialize the memory if requested
        if zeroClear {
            memset(pointer, 0, alignedBytes)
        }

        // Create MLMultiArray with aligned memory
        let array = try MLMultiArray(
            dataPointer: pointer,
            shape: shape,
            dataType: dataType,
            strides: strides,
            deallocator: { bytes in
                // `posix_memalign` requires `free` for cleanup; `deallocate()` would trap.
                Darwin.free(bytes)
            }
        )

        return array
    }

    /// Calculate optimal strides for ANE tile processing
    public static func calculateOptimalStrides(for shape: [NSNumber]) -> [NSNumber] {
        var strides: [Int] = []
        var currentStride = 1

        // Calculate strides from last dimension to first
        for i in (0..<shape.count).reversed() {
            strides.insert(currentStride, at: 0)
            let dimSize = shape[i].intValue

            // Align dimension stride to ANE tile boundaries when beneficial
            if i == shape.count - 1 && dimSize % aneTileSize != 0 {
                // Pad the innermost dimension to tile boundary
                let paddedSize = ((dimSize + aneTileSize - 1) / aneTileSize) * aneTileSize
                currentStride *= paddedSize
            } else {
                currentStride *= dimSize
            }
        }

        return strides.map { NSNumber(value: $0) }
    }

    /// Get element size in bytes for a given data type
    public static func getElementSize(for dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .float16:
            return 2
        case .float32, .float:
            return 4
        case .float64, .double:
            return 8
        case .int32:
            return MemoryLayout<Int32>.stride
        #if swift(>=6.2)
        case .int8:
            return MemoryLayout<Int8>.stride
        #endif
        @unknown default:
            return MemoryLayout<Float>.stride
        }
    }
}

@available(macOS 13.0, iOS 16.0, *)
public final class ANEMemoryOptimizer {
    public static let aneAlignment = ANEMemoryUtils.aneAlignment
    public static let aneTileSize = ANEMemoryUtils.aneTileSize

    private var bufferPool: [String: MLMultiArray] = [:]
    private let bufferLock = NSLock()

    public init() {}

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        return try ANEMemoryUtils.createAlignedArray(
            shape: shape,
            dataType: dataType,
            zeroClear: true
        )
    }

    /// Get or create a reusable buffer from pool
    public func getPooledBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        bufferLock.lock()
        defer { bufferLock.unlock() }

        if let existing = bufferPool[key] {
            // Verify shape matches
            if existing.shape == shape && existing.dataType == dataType {
                return existing
            }
        }

        // Create new buffer
        let buffer = try createAlignedArray(shape: shape, dataType: dataType)
        bufferPool[key] = buffer
        return buffer
    }

    /// Copy data using ANE-optimized memory operations
    public func optimizedCopy<C>(
        from source: C,
        to destination: MLMultiArray,
        offset: Int = 0
    ) where C: Collection, C.Element == Float {
        guard destination.dataType == .float32 else { return }

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(source.count, destination.count - offset)

        // If source is contiguous array, use optimized copy
        if let array = source as? [Float] {
            array.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else if let slice = source as? ArraySlice<Float> {
            // ArraySlice may share contiguous memory
            slice.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else {
            // Fallback for other collections
            var destIndex = offset
            for element in source.prefix(count) {
                destPtr[destIndex] = element
                destIndex += 1
            }
        }
    }

    /// Clear buffer pool to free memory
    public func clearBufferPool() {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        bufferPool.removeAll()
    }
}

@available(macOS 13.0, iOS 16.0, *)
public class ZeroCopyDiarizerFeatureProvider: NSObject, MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    public init(features: [String: MLFeatureValue]) {
        self.features = features
        super.init()
    }

    public var featureNames: Set<String> {
        Set(features.keys)
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }
}

@objc public class VADSegment: NSObject {
    @objc public var start: Double
    @objc public var end: Double

    @objc public init(start: Double, end: Double) {
        self.start = start
        self.end = end
    }
}

private struct WavInfo {
    let sampleRate: Int
    let channels: Int
    let bitsPerSample: Int
    let audioFormat: Int
    let dataOffset: Int64
    let dataSize: Int64
}

private struct WavParseError: Error, LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private struct BorrowedAudioChunk {
    let startIndex: Int
    let sampleCount: Int
    let chunkOffset: Double
}

private func parseWavHeader(fd: Int32) throws -> WavInfo {
    var statBuf = stat()
    guard fstat(fd, &statBuf) == 0 else {
        throw WavParseError(message: "Failed to stat WAV file")
    }
    let fileSize = Int64(statBuf.st_size)
    guard fileSize >= 12 else {
        throw WavParseError(message: "Invalid WAV file (too small)")
    }

    func readBytes(offset: Int64, count: Int) throws -> [UInt8] {
        var buffer = [UInt8](repeating: 0, count: count)
        let bytesRead = buffer.withUnsafeMutableBytes { rawBuffer -> Int in
            guard let base = rawBuffer.baseAddress else { return 0 }
            let readCount = pread(fd, base, count, offset)
            return max(0, readCount)
        }
        if bytesRead != count {
            throw WavParseError(message: "Failed to read WAV header")
        }
        return buffer
    }

    let header = try readBytes(offset: 0, count: 12)
    let riff = String(bytes: header[0..<4], encoding: .ascii)
    let wave = String(bytes: header[8..<12], encoding: .ascii)
    guard riff == "RIFF", wave == "WAVE" else {
        throw WavParseError(message: "Invalid WAV header (missing RIFF/WAVE)")
    }

    func readUInt16LE(_ bytes: [UInt8], _ offset: Int) -> Int {
        let lo = Int(bytes[offset])
        let hi = Int(bytes[offset + 1]) << 8
        return lo | hi
    }

    func readUInt32LE(_ bytes: [UInt8], _ offset: Int) -> Int {
        let b0 = Int(bytes[offset])
        let b1 = Int(bytes[offset + 1]) << 8
        let b2 = Int(bytes[offset + 2]) << 16
        let b3 = Int(bytes[offset + 3]) << 24
        return b0 | b1 | b2 | b3
    }

    var fmtFound = false
    var dataFound = false
    var audioFormat = 0
    var channels = 0
    var sampleRate = 0
    var bitsPerSample = 0
    var dataOffset: Int64 = 0
    var dataSize: Int64 = 0

    var offset: Int64 = 12
    while offset + 8 <= fileSize {
        let chunkHeader = try readBytes(offset: offset, count: 8)
        let chunkId = String(bytes: chunkHeader[0..<4], encoding: .ascii) ?? ""
        let chunkSize = Int64(readUInt32LE(chunkHeader, 4))
        let chunkDataOffset = offset + 8

        if chunkId == "fmt " {
            let fmtSize = Int(min(chunkSize, 16))
            let fmt = try readBytes(offset: chunkDataOffset, count: fmtSize)
            audioFormat = readUInt16LE(fmt, 0)
            channels = readUInt16LE(fmt, 2)
            sampleRate = readUInt32LE(fmt, 4)
            bitsPerSample = readUInt16LE(fmt, 14)
            fmtFound = true
        } else if chunkId == "data" {
            dataOffset = chunkDataOffset
            dataSize = chunkSize
            dataFound = true
        }

        let paddedSize = chunkSize + (chunkSize % 2)
        offset += 8 + paddedSize

        if fmtFound && dataFound {
            break
        }
    }

    guard fmtFound, dataFound else {
        throw WavParseError(message: "Invalid WAV file (missing fmt or data chunk)")
    }

    if dataOffset + dataSize > fileSize {
        throw WavParseError(message: "Invalid WAV data chunk size")
    }

    return WavInfo(
        sampleRate: sampleRate,
        channels: channels,
        bitsPerSample: bitsPerSample,
        audioFormat: audioFormat,
        dataOffset: dataOffset,
        dataSize: dataSize
    )
}

@available(macOS 13.0, iOS 16.0, *)
@objc public class VADProcessor: NSObject {
    private var segmentationModel: MLModel?
    private let chunkSize = 160000  // 10 seconds at 16kHz
    private let sampleRate = 16000
    private let memoryOptimizer = ANEMemoryOptimizer()

    // Configurable parameters
    @objc public var minDurationOn: Double = 0.25   // Minimum speech duration (seconds)
    @objc public var minDurationOff: Double = 0.1   // Minimum gap before merging (seconds)

    @objc public override init() {
        super.init()
    }

    // Initialize with model path
    @objc public func loadModel(at path: String) -> Bool {
        do {
            let modelURL = URL(fileURLWithPath: path)
            let config = MLModelConfiguration()

            // Use CPU+ANE for optimal performance
            config.computeUnits = .cpuAndNeuralEngine
            config.allowLowPrecisionAccumulationOnGPU = true

            // Load and compile the model
            self.segmentationModel = try MLModel(contentsOf: modelURL, configuration: config)

            // Warm up the model with a dummy prediction to ensure ANE is ready
            warmupModel()

            return true
        } catch {
            print("Error loading model: \(error)")
            return false
        }
    }

    // Warm up the model to ensure ANE is ready
    private func warmupModel() {
        guard let model = segmentationModel else { return }

        do {
            // Create dummy input
            let dummyArray = try memoryOptimizer.createAlignedArray(
                shape: [1, 1, NSNumber(value: chunkSize)],
                dataType: .float32
            )

            let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
                "audio": MLFeatureValue(multiArray: dummyArray)
            ])

            // Run dummy prediction to warm up ANE
            _ = try? model.prediction(from: featureProvider)
        } catch {
            print("Warmup failed: \(error)")
        }
    }

    // Process audio file and return segments
    @objc public func processAudioFile(at path: String) -> [VADSegment] {
        guard let model = segmentationModel else {
            print("Model not loaded")
            return []
        }

        do {
            return try processStreamedWav(at: path, model: model)
        } catch {
            print("Failed to process audio file: \(error)")
            return []
        }
    }

    public func processAudioSamples(_ samplesPtr: UnsafePointer<Float>, sampleCount: Int) -> [VADSegment] {
        guard let model = segmentationModel else {
            print("Model not loaded")
            return []
        }

        guard sampleCount > 0 else {
            return []
        }

        var allSegments: [VADSegment] = []
        let batchSize = 64
        var batch: [BorrowedAudioChunk] = []
        batch.reserveCapacity(batchSize)

        var offsetSamples = 0
        while offsetSamples < sampleCount {
            let samplesToRead = min(chunkSize, sampleCount - offsetSamples)
            if samplesToRead <= 0 {
                break
            }

            let chunkOffsetTime = Double(offsetSamples) / Double(sampleRate)
            batch.append(BorrowedAudioChunk(
                startIndex: offsetSamples,
                sampleCount: samplesToRead,
                chunkOffset: chunkOffsetTime
            ))

            if batch.count >= batchSize {
                let batchSegments = processBatch(samplesPtr: samplesPtr, batch: batch, model: model)
                allSegments.append(contentsOf: batchSegments)
                batch.removeAll(keepingCapacity: true)
            }

            offsetSamples += samplesToRead
        }

        if !batch.isEmpty {
            let batchSegments = processBatch(samplesPtr: samplesPtr, batch: batch, model: model)
            allSegments.append(contentsOf: batchSegments)
        }

        return mergeSegments(allSegments)
    }

    private func processStreamedWav(at path: String, model: MLModel) throws -> [VADSegment] {
        let fd = open(path, O_RDONLY)
        if fd < 0 {
            throw WavParseError(message: "Failed to open audio file at \(path)")
        }
        defer {
            _ = close(fd)
        }

        let info = try parseWavHeader(fd: fd)
        if info.sampleRate != sampleRate {
            throw WavParseError(message: "Unsupported WAV sample rate \(info.sampleRate)Hz; expected \(sampleRate)Hz")
        }
        if info.channels != 1 || info.bitsPerSample != 16 || info.audioFormat != 1 {
            throw WavParseError(message: "Unsupported WAV format: require 16-bit PCM mono")
        }

        let bytesPerSample = info.bitsPerSample / 8
        let totalSamples = Int(info.dataSize / Int64(bytesPerSample))
        if totalSamples <= 0 {
            return []
        }

        var allSegments: [VADSegment] = []
        let batchSize = 64
        var batch: [(ArraySlice<Float>, Double)] = []
        batch.reserveCapacity(batchSize)

        let scale = Float(1.0 / 32768.0)
        var offsetSamples = 0

        while offsetSamples < totalSamples {
            let samplesToRead = min(chunkSize, totalSamples - offsetSamples)
            if samplesToRead <= 0 {
                break
            }

            var int16Buffer = [Int16](repeating: 0, count: samplesToRead)
            let byteOffset = info.dataOffset + Int64(offsetSamples * bytesPerSample)
            let byteCount = samplesToRead * bytesPerSample

            let bytesRead = int16Buffer.withUnsafeMutableBytes { rawBuffer -> Int in
                guard let base = rawBuffer.baseAddress else { return 0 }
                var totalRead = 0
                while totalRead < byteCount {
                    let readCount = pread(
                        fd,
                        base.advanced(by: totalRead),
                        byteCount - totalRead,
                        byteOffset + Int64(totalRead)
                    )
                    if readCount <= 0 {
                        break
                    }
                    totalRead += readCount
                }
                return totalRead
            }

            let samplesRead = max(0, min(samplesToRead, bytesRead / bytesPerSample))
            if samplesRead == 0 {
                break
            }

            var floatBuffer = [Float](repeating: 0, count: samplesRead)
            for i in 0..<samplesRead {
                floatBuffer[i] = Float(int16Buffer[i]) * scale
            }

            let chunkOffsetTime = Double(offsetSamples) / Double(sampleRate)
            batch.append((floatBuffer[0..<floatBuffer.count], chunkOffsetTime))

            if batch.count >= batchSize {
                let batchSegments = processBatch(batch, model: model)
                allSegments.append(contentsOf: batchSegments)
                batch.removeAll(keepingCapacity: true)
            }

            offsetSamples += samplesRead
        }

        if !batch.isEmpty {
            let batchSegments = processBatch(batch, model: model)
            allSegments.append(contentsOf: batchSegments)
        }

        return mergeSegments(allSegments)
    }

    // Process a batch of chunks concurrently
    private func processBatch(_ batch: [(ArraySlice<Float>, Double)], model: MLModel) -> [VADSegment] {
        var allSegments: [VADSegment] = []
        let semaphore = DispatchSemaphore(value: 0)
        let queue = DispatchQueue.global(qos: .userInitiated)
        var results: [[VADSegment]?] = Array(repeating: nil, count: batch.count)

        // Process each chunk in the batch concurrently
        for (index, (chunk, chunkOffset)) in batch.enumerated() {
            queue.async {
                do {
                    let segments = try self.processChunkOptimized(chunk, model: model, chunkOffset: chunkOffset)
                    results[index] = segments
                } catch {
                    print("Error processing chunk: \(error)")
                    results[index] = []
                }
                semaphore.signal()
            }
        }

        // Wait for all chunks to complete
        for _ in 0..<batch.count {
            semaphore.wait()
        }

        // Collect results in order
        for segments in results {
            if let segments = segments {
                allSegments.append(contentsOf: segments)
            }
        }

        return allSegments
    }

    // Process a batch of borrowed chunks concurrently
    private func processBatch(samplesPtr: UnsafePointer<Float>, batch: [BorrowedAudioChunk], model: MLModel) -> [VADSegment] {
        var allSegments: [VADSegment] = []
        let semaphore = DispatchSemaphore(value: 0)
        let queue = DispatchQueue.global(qos: .userInitiated)
        var results: [[VADSegment]?] = Array(repeating: nil, count: batch.count)

        // Process each chunk in the batch concurrently
        for (index, chunk) in batch.enumerated() {
            queue.async {
                do {
                    let segments = try self.processChunkOptimized(samplesPtr: samplesPtr, chunk: chunk, model: model)
                    results[index] = segments
                } catch {
                    print("Error processing chunk: \(error)")
                    results[index] = []
                }
                semaphore.signal()
            }
        }

        // Wait for all chunks to complete
        for _ in 0..<batch.count {
            semaphore.wait()
        }

        // Collect results in order
        for segments in results {
            if let segments = segments {
                allSegments.append(contentsOf: segments)
            }
        }

        return allSegments
    }

    // Optimized chunk processing with buffer reuse
    private func processChunkOptimized(_ audioChunk: ArraySlice<Float>, model: MLModel, chunkOffset: Double) throws -> [VADSegment] {
        // Get pooled buffer for this thread
        let threadId = Thread.current.description
        let bufferKey = "audio_buffer_\(threadId)"

        // Get or create ANE-aligned buffer from pool
        let audioArray = try memoryOptimizer.getPooledBuffer(
            key: bufferKey,
            shape: [1, 1, NSNumber(value: chunkSize)],
            dataType: .float32
        )

        // Clear buffer first (important for reuse)
        let ptr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        memset(ptr, 0, chunkSize * MemoryLayout<Float>.size)

        // Copy audio data
        let copyCount = min(audioChunk.count, chunkSize)
        audioChunk.prefix(chunkSize).withUnsafeBufferPointer { buffer in
            vDSP_mmov(
                buffer.baseAddress!,
                ptr,
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
        }

        // Create zero-copy feature provider
        let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
            "audio": MLFeatureValue(multiArray: audioArray)
        ])

        // Configure prediction for ANE
        let options = MLPredictionOptions()

        // Use async prediction if available for better ANE scheduling
        let output: MLFeatureProvider
        if #available(macOS 14.0, iOS 17.0, *) {
            // Prefetch to ANE
            audioArray.prefetchToNeuralEngine()
            // Use async for better scheduling
            output = try model.prediction(from: featureProvider, options: options)
        } else {
            output = try model.prediction(from: featureProvider, options: options)
        }

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            return []
        }

        // Process segments with optimized memory access
        return processSegmentsOptimized(segmentOutput, chunkOffset: chunkOffset)
    }

    // Optimized chunk processing with buffer reuse
    private func processChunkOptimized(samplesPtr: UnsafePointer<Float>, chunk: BorrowedAudioChunk, model: MLModel) throws -> [VADSegment] {
        // Get pooled buffer for this thread
        let threadId = Thread.current.description
        let bufferKey = "audio_buffer_\(threadId)"

        // Get or create ANE-aligned buffer from pool
        let audioArray = try memoryOptimizer.getPooledBuffer(
            key: bufferKey,
            shape: [1, 1, NSNumber(value: chunkSize)],
            dataType: .float32
        )

        // Clear buffer first (important for reuse)
        let ptr = audioArray.dataPointer.assumingMemoryBound(to: Float.self)
        memset(ptr, 0, chunkSize * MemoryLayout<Float>.size)

        // Borrow from the caller-owned waveform and only copy the current chunk
        let copyCount = min(chunk.sampleCount, chunkSize)
        if copyCount > 0 {
            let sourcePtr = samplesPtr.advanced(by: chunk.startIndex)
            vDSP_mmov(
                sourcePtr,
                ptr,
                vDSP_Length(copyCount),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(copyCount)
            )
        }

        // Create zero-copy feature provider
        let featureProvider = ZeroCopyDiarizerFeatureProvider(features: [
            "audio": MLFeatureValue(multiArray: audioArray)
        ])

        // Configure prediction for ANE
        let options = MLPredictionOptions()

        // Use async prediction if available for better ANE scheduling
        let output: MLFeatureProvider
        if #available(macOS 14.0, iOS 17.0, *) {
            // Prefetch to ANE
            audioArray.prefetchToNeuralEngine()
            // Use async for better scheduling
            output = try model.prediction(from: featureProvider, options: options)
        } else {
            output = try model.prediction(from: featureProvider, options: options)
        }

        guard let segmentOutput = output.featureValue(for: "segments")?.multiArrayValue else {
            return []
        }

        // Process segments with optimized memory access
        return processSegmentsOptimized(segmentOutput, chunkOffset: chunk.chunkOffset)
    }

    private func processSegmentsOptimized(_ segmentOutput: MLMultiArray, chunkOffset: Double) -> [VADSegment] {
        let frames = segmentOutput.shape[1].intValue
        let combinations = segmentOutput.shape[2].intValue

        // Pre-allocate result array
        var segments = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: combinations),
                count: frames),
            count: 1)

        // Use direct memory access for better performance
        let ptr = segmentOutput.dataPointer.assumingMemoryBound(to: Float.self)

        // Copy data in a cache-friendly manner
        for f in 0..<frames {
            segments[0][f].withUnsafeMutableBufferPointer { buffer in
                vDSP_mmov(
                    ptr.advanced(by: f * combinations),
                    buffer.baseAddress!,
                    vDSP_Length(combinations),
                    1,
                    vDSP_Length(combinations),
                    1
                )
            }
        }

        // Apply powerset conversion
        return powersetConversionOptimized(segments, chunkOffset: chunkOffset)
    }

    private func powersetConversionOptimized(_ segments: [[[Float]]], chunkOffset: Double) -> [VADSegment] {
        let powerset: [[Int]] = [
            [],        // 0: silence
            [0],       // 1: speaker 0
            [1],       // 2: speaker 1
            [2],       // 3: speaker 2
            [0, 1],    // 4: speakers 0 & 1
            [0, 2],    // 5: speakers 0 & 2
            [1, 2],    // 6: speakers 1 & 2
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        // Use ANE-aligned array for result
        let binarizedArray = try? memoryOptimizer.createAlignedArray(
            shape: [batchSize, numFrames, numSpeakers] as [NSNumber],
            dataType: .float32
        )

        guard let binarizedArray = binarizedArray else {
            // Fallback to regular array
            return powersetConversionFallback(segments, chunkOffset: chunkOffset)
        }

        // Direct memory access
        let ptr = binarizedArray.dataPointer.assumingMemoryBound(to: Float.self)

        // Process all frames
        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                // Find max using vDSP
                var maxValue: Float = 0
                var maxIndex: vDSP_Length = 0
                frame.withUnsafeBufferPointer { buffer in
                    vDSP_maxvi(buffer.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(frame.count))
                }

                // Set speakers based on powerset
                let baseIdx = (b * numFrames + f) * numSpeakers
                for speaker in powerset[Int(maxIndex)] {
                    ptr[baseIdx + speaker] = 1.0
                }
            }
        }

        // Convert to VAD segments (any speaker activity = speech)
        var vadFrames = [Bool](repeating: false, count: numFrames)

        for f in 0..<numFrames {
            var hasActivity = false
            for s in 0..<numSpeakers {
                let idx = f * numSpeakers + s
                if ptr[idx] > 0.5 {
                    hasActivity = true
                    break
                }
            }
            vadFrames[f] = hasActivity
        }

        // Convert frames to time segments
        let frameStep = 0.016875  // From pyannote model
        var segments: [VADSegment] = []
        var inSpeech = false
        var segmentStart = 0.0

        for (idx, isActive) in vadFrames.enumerated() {
            let currentTime = chunkOffset + Double(idx) * frameStep

            if isActive && !inSpeech {
                inSpeech = true
                segmentStart = currentTime
            } else if !isActive && inSpeech {
                inSpeech = false
                segments.append(VADSegment(start: segmentStart, end: currentTime))
            }
        }

        // Close final segment if needed
        if inSpeech {
            let endTime = chunkOffset + Double(numFrames) * frameStep
            segments.append(VADSegment(start: segmentStart, end: endTime))
        }

        return segments
    }

    private func powersetConversionFallback(_ segments: [[[Float]]], chunkOffset: Double) -> [VADSegment] {
        // Original implementation as fallback
        let powerset: [[Int]] = [
            [],
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
        ]

        let batchSize = segments.count
        let numFrames = segments[0].count
        let numSpeakers = 3

        var binarized = Array(
            repeating: Array(
                repeating: Array(repeating: 0.0 as Float, count: numSpeakers),
                count: numFrames
            ),
            count: batchSize
        )

        for b in 0..<batchSize {
            for f in 0..<numFrames {
                let frame = segments[b][f]

                guard let bestIdx = frame.indices.max(by: { frame[$0] < frame[$1] }) else {
                    continue
                }

                for speaker in powerset[bestIdx] {
                    binarized[b][f][speaker] = 1.0
                }
            }
        }

        // Convert to VAD
        var vadFrames = [Bool](repeating: false, count: numFrames)
        for f in 0..<numFrames {
            vadFrames[f] = binarized[0][f].contains { $0 > 0.5 }
        }

        // Convert to time segments
        let frameStep = 0.016875
        var segments: [VADSegment] = []
        var inSpeech = false
        var segmentStart = 0.0

        for (idx, isActive) in vadFrames.enumerated() {
            let currentTime = chunkOffset + Double(idx) * frameStep

            if isActive && !inSpeech {
                inSpeech = true
                segmentStart = currentTime
            } else if !isActive && inSpeech {
                inSpeech = false
                segments.append(VADSegment(start: segmentStart, end: currentTime))
            }
        }

        if inSpeech {
            let endTime = chunkOffset + Double(numFrames) * frameStep
            segments.append(VADSegment(start: segmentStart, end: endTime))
        }

        return segments
    }

    private func mergeSegments(_ segments: [VADSegment]) -> [VADSegment] {
        guard !segments.isEmpty else { return [] }

        let sorted = segments.sorted { $0.start < $1.start }
        var merged: [VADSegment] = []
        var current = sorted[0]

        for i in 1..<sorted.count {
            let next = sorted[i]
            // Use minDurationOff as gap threshold
            if next.start - current.end <= minDurationOff {
                // Merge segments
                current = VADSegment(start: current.start, end: max(current.end, next.end))
            } else {
                merged.append(current)
                current = next
            }
        }
        merged.append(current)

        // Filter out segments shorter than minDurationOn
        return merged.filter { $0.end - $0.start >= minDurationOn }
    }
}

@available(macOS 14.0, iOS 17.0, *)
extension MLMultiArray {
    /// Prefetch data to Neural Engine
    public func prefetchToNeuralEngine() {
        // Trigger ANE prefetch by accessing first and last elements
        if count > 0 {
            _ = self[0]
            _ = self[count - 1]
        }
    }
}

@_cdecl("vad_create")
public func vad_create() -> UnsafeMutableRawPointer {
    let processor = VADProcessor()
    return Unmanaged.passRetained(processor).toOpaque()
}

@_cdecl("vad_load_model")
public func vad_load_model(_ processorPtr: UnsafeMutableRawPointer, _ modelPath: UnsafePointer<CChar>) -> Int32 {
    let processor = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeUnretainedValue()
    let path = String(cString: modelPath)
    return processor.loadModel(at: path) ? 1 : 0
}

@_cdecl("vad_set_min_duration_on")
public func vad_set_min_duration_on(_ processorPtr: UnsafeMutableRawPointer, _ value: Double) {
    let processor = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeUnretainedValue()
    processor.minDurationOn = value
}

@_cdecl("vad_set_min_duration_off")
public func vad_set_min_duration_off(_ processorPtr: UnsafeMutableRawPointer, _ value: Double) {
    let processor = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeUnretainedValue()
    processor.minDurationOff = value
}

@_cdecl("vad_process_file")
public func vad_process_file(
    _ processorPtr: UnsafeMutableRawPointer,
    _ audioPath: UnsafePointer<CChar>,
    _ count: UnsafeMutablePointer<Int32>
) -> UnsafeMutableRawPointer {
    let processor = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeUnretainedValue()
    let path = String(cString: audioPath)

    let segments = processor.processAudioFile(at: path)
    count.pointee = Int32(segments.count)

    guard segments.count > 0 else {
        return UnsafeMutableRawPointer(bitPattern: 1)!  // Non-null but invalid pointer
    }

    // Allocate memory for doubles (start, end pairs)
    let buffer = UnsafeMutablePointer<Double>.allocate(capacity: segments.count * 2)
    for (i, seg) in segments.enumerated() {
        buffer[i * 2] = seg.start
        buffer[i * 2 + 1] = seg.end
    }

    return UnsafeMutableRawPointer(buffer)
}

@_cdecl("vad_process_samples")
public func vad_process_samples(
    _ processorPtr: UnsafeMutableRawPointer,
    _ samplesPtr: UnsafePointer<Float>,
    _ sampleCount: Int,
    _ count: UnsafeMutablePointer<Int32>
) -> UnsafeMutableRawPointer {
    let processor = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeUnretainedValue()
    let segments = processor.processAudioSamples(samplesPtr, sampleCount: sampleCount)
    count.pointee = Int32(segments.count)

    guard segments.count > 0 else {
        return UnsafeMutableRawPointer(bitPattern: 1)!
    }

    let buffer = UnsafeMutablePointer<Double>.allocate(capacity: segments.count * 2)
    for (i, seg) in segments.enumerated() {
        buffer[i * 2] = seg.start
        buffer[i * 2 + 1] = seg.end
    }

    return UnsafeMutableRawPointer(buffer)
}

@_cdecl("vad_free_segments")
public func vad_free_segments(_ segments: UnsafeMutableRawPointer) {
    // Check for our special invalid pointer
    if segments == UnsafeMutableRawPointer(bitPattern: 1) {
        return
    }
    segments.deallocate()
}

@_cdecl("vad_destroy")
public func vad_destroy(_ processorPtr: UnsafeMutableRawPointer) {
    _ = Unmanaged<VADProcessor>.fromOpaque(processorPtr).takeRetainedValue()
}
