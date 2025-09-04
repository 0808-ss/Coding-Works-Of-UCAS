echo "INFO: Compiling design using build_2.do..."
do build_2.do
echo "INFO: Compilation complete."

echo "INFO: Loading simulation for tb_top_2..."
vsim +nowarnTSCALE -lib work -voptargs=+acc tb_top_2
echo "INFO: Simulation loaded into GUI."

echo "INFO: Adding specific signals to wave window..."
add wave /tb_top_2/result

echo "INFO: Signals added to wave window."

echo "INFO: Running simulation (run -all)..."
run -all
echo "INFO: Simulation finished. Waveform is now available in the wave window."