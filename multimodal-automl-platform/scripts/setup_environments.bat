@echo off
echo [1/4] Khoi tao Python Virtual Environment (automl-core)...
cd automl-core
python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

echo [2/4] Cai dat Node.js Gateway dependencies...
cd gateway-orchestrator
call npm install
cd ..

echo [3/4] Kiem tra cau truc thu muc DVC data...
if not exist data\raw\images mkdir data\raw\images
if not exist data\raw\tabular mkdir data\raw\tabular
if not exist data\processed mkdir data\processed
if not exist data\artifacts mkdir data\artifacts

echo [4/4] Bien dich C++ Edge Inference Core...
cd edge-inference-cpp
if not exist build mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..\..
echo Hoan tat cai dat moi truong!
pause
