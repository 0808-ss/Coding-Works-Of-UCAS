echo "INFO: Compiling design using build_0.do..."
do build_0.do
echo "INFO: Compilation complete."

echo "INFO: Loading simulation for tb_top_0..."
vsim +nowarnTSCALE -lib work -voptargs=+acc tb_top_0
echo "INFO: Simulation loaded into GUI."

echo "INFO: Adding specific signals to wave window..."
add wave /tb_top_0/result
add wave /tb_top_0/pe_ctl
add wave /tb_top_0/pe_result

echo "INFO: Signals added to wave window."

echo "INFO: Running simulation (run -all)..."
run -all
echo "INFO: Simulation finished. Waveform is now available in the wave window."