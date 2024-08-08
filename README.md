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
brew install cmake ninja opencv libomp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -zxvf onnxruntime-osx-arm64-1.16.3.tgz

# for Release build
cmake -S. -BRelease -DBUILD_DEMO=ON -DBUILD_TEST=ON
cmake --build Release -j

# for Debug build
cmake -S. -BDebug -DBUILD_DEMO=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build Debug -j
```

## 2.3. Linux (x86-64)

```bash
git clone https://github.com/angseung/pcdet-infer-cpu.git
cd pcdet-infer-cpu
sudo apt update
sudo apt install cmake libopencv-dev build-essential -y
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

## 3.2. demo

- This program runs a PCDet model with PCD files, and then draws detected bounding boxes on point cloud in
  bird-eyes-view.

```bash
./Release/bin/demo ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

### 3.2.1. demo_c

- This program has same functions with `demo`, but it is implemented with inference codes
  in `pcdet_c.h` & `pcdet_c.cpp`.
- Refer this code if you want to use this repository in C project.

```bash
./Release/bin/demo_c ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

---

# Update History

## Release 1.1.0

### 1. Update

- Removed Point Cloud Library from dependency.
    - We manually implemented a library for parsing points from pcd files.
- Optimized point shuffle process.
- Renamed `PCDet` class to `PCDetCPU`.
- Implemented base abstact `PCDet` & `Model` class for `PCDetCPU` & `OrtModel` class respectively.
- Support `<<` operator for `Runtimeconfig` & `Metadata` class
    - This operator prints all member variables in `Runtimeconfig` & `Metadata`.
- Modified `Point` structure with template structure.

### 2. Bug Fix

- Fixed assertion error that can be occured if point value is on grid edge.
- Removed `MAX_POINT_NUM` from a pcd parser.
- Modified all `int&` and `float&` type arguments to `int` and `float` .

---

## Release 1.2.0

### 1. Update

- Class label are modified from 1, 2, 3, … to 0, 1, 2, …
- Added assertion to check `CLASS_NUM == IOU_RECTIFIER.size()`
- Removed all C style casting for more stable code

---

## Release 1.3.0

### 1. Bug Fix

- Fixed segmentation fault error that can occur if `MAX_VOXELS` in model configuration is smaller than the number of
  voxels in input data.

### 2. Update

- Support language binding for C.
    - Source codes are available in `pcdet_c.cpp` & `pcdet_c.h`.

---
