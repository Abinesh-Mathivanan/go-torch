// implements the safetensors file format: an 8-byte little-endian header length, a JSON header describing each tensor's
// byte-range, followed by the raw tensor bytes. 
// Only the F64 dtype is supported, since go-torch's Tensor stores everything as float64

package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"go-torch/tensor"
	"io"
	"math"
	"os"
	"sort"
)

// dtypeF64 is the only dtype this package currently reads or writes.
const dtypeF64 = "F64"

// maxHeaderSize bounds how large a header we're willing to allocate for
// before we've validated anything in it
const maxHeaderSize = 100 * 1024 * 1024

// tensorInfo mirrors one entry of the safetensors header JSON.
type tensorInfo struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Save writes tensors (name -> Tensor) to path in safetensors format.
// metadata is optional arbitrary string key/value pairs stored under the
// format's reserved "__metadata__" header key (e.g. {"format":
// "go-torch-v1"}); pass nil if you don't need any.
func Save(path string, tensors map[string]*tensor.Tensor, metadata map[string]string) error {
	if len(tensors) == 0 {
		return fmt.Errorf("safetensors: cannot save an empty tensor map")
	}

	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	header := make(map[string]interface{}, len(names)+1)
	var dataBuf bytes.Buffer
	var offset int64

	for _, name := range names {
		t := tensors[name]
		if t == nil {
			return fmt.Errorf("safetensors: tensor %q is nil", name)
		}
		data := t.GetData()
		byteLen := int64(len(data)) * 8

		buf := make([]byte, byteLen)
		for i, v := range data {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		dataBuf.Write(buf)

		header[name] = tensorInfo{
			DType:       dtypeF64,
			Shape:       t.GetShape(),
			DataOffsets: [2]int64{offset, offset + byteLen},
		}
		offset += byteLen
	}

	if metadata != nil {
		header["__metadata__"] = metadata
	}

	headerJSON, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("safetensors: failed to encode header: %w", err)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("safetensors: failed to create file %s: %w", path, err)
	}
	defer file.Close()

	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := file.Write(lenBuf[:]); err != nil {
		return fmt.Errorf("safetensors: failed to write header length to %s: %w", path, err)
	}
	if _, err := file.Write(headerJSON); err != nil {
		return fmt.Errorf("safetensors: failed to write header to %s: %w", path, err)
	}
	if _, err := file.Write(dataBuf.Bytes()); err != nil {
		return fmt.Errorf("safetensors: failed to write tensor data to %s: %w", path, err)
	}

	return nil
}

// Load reads a safetensors file and returns its tensors by name, plus any
// metadata stored under "__metadata__".
func Load(path string) (map[string]*tensor.Tensor, map[string]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: failed to open file %s: %w", path, err)
	}
	defer file.Close()

	var lenBuf [8]byte
	if _, err := io.ReadFull(file, lenBuf[:]); err != nil {
		return nil, nil, fmt.Errorf("safetensors: failed to read header length from %s: %w", path, err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen == 0 || headerLen > maxHeaderSize {
		return nil, nil, fmt.Errorf("safetensors: implausible header length %d in %s", headerLen, path)
	}

	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(file, headerBytes); err != nil {
		return nil, nil, fmt.Errorf("safetensors: failed to read header from %s: %w", path, err)
	}

	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, nil, fmt.Errorf("safetensors: failed to parse header JSON from %s: %w", path, err)
	}

	var metadata map[string]string
	if raw, ok := rawHeader["__metadata__"]; ok {
		if err := json.Unmarshal(raw, &metadata); err != nil {
			return nil, nil, fmt.Errorf("safetensors: failed to parse __metadata__ in %s: %w", path, err)
		}
		delete(rawHeader, "__metadata__")
	}

	// Read the rest of the file in one go so every declared offset can be
	// bounds-checked against the real data length before any of them are
	// trusted for a slice operation.
	dataBytes, err := io.ReadAll(file)
	if err != nil {
		return nil, nil, fmt.Errorf("safetensors: failed to read tensor data from %s: %w", path, err)
	}

	result := make(map[string]*tensor.Tensor, len(rawHeader))
	for name, raw := range rawHeader {
		var info tensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, nil, fmt.Errorf("safetensors: failed to parse tensor entry %q in %s: %w", name, path, err)
		}
		if info.DType != dtypeF64 {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %q (only %q is supported)", name, info.DType, dtypeF64)
		}

		begin, end := info.DataOffsets[0], info.DataOffsets[1]
		if begin < 0 || end < begin || end > int64(len(dataBytes)) {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has invalid data_offsets [%d, %d] for a %d-byte data section", name, begin, end, len(dataBytes))
		}

		numel := 1
		for _, d := range info.Shape {
			if d <= 0 {
				return nil, nil, fmt.Errorf("safetensors: tensor %q has non-positive shape dimension in %v", name, info.Shape)
			}
			numel *= d
		}
		expectedBytes := int64(numel) * 8
		if end-begin != expectedBytes {
			return nil, nil, fmt.Errorf("safetensors: tensor %q shape %v implies %d bytes but data_offsets span %d bytes", name, info.Shape, expectedBytes, end-begin)
		}

		raw := dataBytes[begin:end]
		values := make([]float64, numel)
		for i := range values {
			bits := binary.LittleEndian.Uint64(raw[i*8:])
			values[i] = math.Float64frombits(bits)
		}

		t, err := tensor.NewTensor(info.Shape, values)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: failed to construct tensor %q: %w", name, err)
		}
		result[name] = t
	}

	return result, metadata, nil
}