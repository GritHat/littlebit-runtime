"""Re-quantizes a Next-Concept gguf to a smaller block format.

The tensors that are Q8_0 in REF (a file written by littlebit-ncm-convert --outtype q8_0) are taken from SRC (the same
model in F32) and stored as TYPE; F16 tensors stay F16 and the rest F32. gguf-py comes from the llama.cpp tree:

    python tools/ncm/requant.py tiny-f32.gguf tiny-q8.gguf tiny-q4.gguf Q4_0 build/_deps/llama_cpp-src/gguf-py
"""
import sys

if len(sys.argv) != 6:
    sys.exit(__doc__)
sys.path.insert(0, sys.argv[5])
import gguf  # noqa: E402
import numpy as np  # noqa: E402

src, ref, out, typ = sys.argv[1:5]
T = getattr(gguf.GGMLQuantizationType, typ)
r, rr = gguf.GGUFReader(src), gguf.GGUFReader(ref)
reftype = {t.name: t.tensor_type for t in rr.tensors}
w = gguf.GGUFWriter(out, arch=str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8"))
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture":
        continue
    vt = f.types[0]
    if vt == gguf.GGUFValueType.STRING:
        w.add_string(k, str(bytes(f.parts[-1]), "utf-8"))
    elif vt == gguf.GGUFValueType.ARRAY:
        if f.types[-1] == gguf.GGUFValueType.STRING:
            w.add_array(k, [str(bytes(f.parts[i]), "utf-8") for i in f.data])
        else:
            w.add_array(k, [f.parts[i].tolist()[0] for i in f.data])
    else:
        w.add_key_value(k, f.parts[-1].tolist()[0], vt)
for t in r.tensors:
    data = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)])
    rt = reftype.get(t.name)
    if rt == gguf.GGMLQuantizationType.Q8_0 and data.shape[-1] % gguf.GGML_QUANT_SIZES[T][0] == 0:
        w.add_tensor(t.name, gguf.quants.quantize(data, T), raw_dtype=T)
    elif rt == gguf.GGMLQuantizationType.F16:
        w.add_tensor(t.name, data.astype(np.float16))
    else:
        w.add_tensor(t.name, data)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("written", out)
