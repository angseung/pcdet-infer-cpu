# 1. PCDet-Infer-CPU

- An Inference library for PCDet models.
- An Inference library for CPU only devices.
    - Supports x86-64 & AArch64
    - Supports macOS & Linux
- GPU Version ONNXRuntime are not supported yet.

# 2. Build

- This repo tested ONLY with onnxruntime==1.16.3.

## 2.1 Dependencies

- GCC or Clang Complier
- OpenCV
- cmake ≥ 3.11

## 2.2. macOS (AArch64)

```bash
git clone https://github.com/angseung/pcdet-infer-cpu.git
cd pcdet-infer-cpu
brew install cmake ninja opencv
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -zxvf onnxruntime-osx-arm64-1.16.3.tgz

# for Release build
cmake -S. -BRelease -DBUILD_DEMO=ON -DBUILD_TEST=ON
cmake --build Release -j

# for Debug build
cmake -S. -BDebug -DBUILD_DEMO=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build Debug -j
```

## 2.3. Linux (x86)

```bash
git clone https://github.com/angseung/pcdet-infer-cpu.git
cd pcdet-infer-cpu
sudo apt update
sudo apt install cmake libopencv-dev -y
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -zxvf onnxruntime-linux-x64-1.16.3.tgz

# for Release build
cmake -S. -BRelease -DBUILD_DEMO=ON -DBUILD_TEST=ON
cmake --build Release -j

# for Debug build
cmake -S. -BDebug -DBUILD_DEMO=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build Debug -j
```

# 3. Run

- `YOUR_PCD_DIR_PATH` means a parent directory of pcd files.
- `YOUR_METADATA_FILE_PATH` means a `metadata.json` file that describes configurations of your PCDet models.

## 3.1. main

- This program runs a PCDet model with PCD files, and then shows the number of detected objects in each classes

```bash
./Release/bin/main ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

## 3.2. Demo

- This program runs a PCDet model with PCD files, and then draws detected bounding boxes on point cloud in
  bird-eyes-view.

```bash
./Release/bin/main ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```
