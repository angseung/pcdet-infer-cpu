# 1. PCDet-Infer-CPU

- An Inference library for PCDet models.
- An Inference library for CPU only devices.
    - Support x86-64 & AArch64
    - Support macOS & Linux
- GPU Version ONNXRuntime is not supported yet.

# 2. Build

- This repo tested ONLY with `onnxruntime`==1.16.3.

## 2.1 Dependencies

- GCC or Clang Complier
- OpenCV
- cmake ≥ 3.24

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
cmake -S. -BDebug -DBUILD_DEMO=ON -DBUILD_TEST=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build Debug -j
```

# 2.2. Linux (x86)

```bash
git clone https://github.com/angseung/pcdet-infer-cpu.git
cd pcdet-infer-cpu
sudo apt update
sudo apt install cmake build-essential libopencv-dev -y
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -zxvf onnxruntime-linux-x64-1.16.3.tgz

# for Release build
cmake -S. -BRelease -DBUILD_DEMO=ON -DBUILD_TEST=OFF
cmake --build Release -j

# for Debug build
cmake -S. -BDebug -DBUILD_DEMO=ON -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build Debug -j
```

# 3. Run

- `YOUR_PCD_DIR_PATH` indicates a parent directory of pcd files.
- `YOUR_METADATA_FILE_PATH` indicates a `metadata.json` file that describes configurations of your PCDet models.

## 3.1. main

- This program runs a PCDet model with PCD files, and then shows the number of detected objects in each classes

```bash
./Release/bin/main ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

## 3.2. demo

### 3.2.1. demo

- This program runs a PCDet model with PCD files, and then draws detected bounding boxes on point cloud in
  bird-eyes-view.

```bash
./Release/bin/demo ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

### 3.2.2. demo_c

- This program has same functions with `demo`, but it is implemented with inference codes
  in `pcdet_c.h` & `pcdet_c.cpp`.
- Please refer `pcdet_c.h` & `pcdet_c.cpp` if you want to use this repository in your C project.

```bash
./Release/bin/demo_c ./YOUR_PCD_DIR_PATH ./YOUR_METADATA_FILE_PATH
```

---

# Update History

## Release 1.0.0

- Initial Release

---

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

## Release 1.4.0

### 1. Update

- Support 3D drawing for PCD file.
    - Source codes are available in `demo_3d.cpp`.

---

## Release 1.4.1

### 1. Update

- Removed `demo_3d.cpp`.
- Added library compression script in `src/pcdet-infer-cpu/CMakeLists.txt`.

---

## Release 1.4.2

### 1. Update

- Renamed `type.h` → `box.h` for a compatible issue.
- Separated `CONF_THRE` for each classes.
    - Removed `CONF_THRE` in `runtimeconfig.h`.
    - Added macros indicate threshold for each classes in `demo_common.h`.
