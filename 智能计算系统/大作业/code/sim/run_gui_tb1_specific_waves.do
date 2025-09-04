echo "INFO: Compiling design using build_1.do..."
do build_1.do
echo "INFO: Compilation complete."

echo "INFO: Loading simulation for tb_top_1..."
vsim +nowarnTSCALE -lib work -voptargs=+acc tb_top_1
echo "INFO: Simulation loaded into GUI."

echo "INFO: Adding specific signals to wave window..."
add wave /tb_top_1/clk
add wave /tb_top_1/rst_n
add wave /tb_top_1/inst_addr
add wave /tb_top_1/neuron_addr
add wave /tb_top_1/weight_addr
add wave /tb_top_1/pe_inst
add wave /tb_top_1/pe_ctl
add wave /tb_top_1/pe_vld_i
add wave /tb_top_1/pe_result
add wave /tb_top_1/pe_vld_o
add wave /tb_top_1/result_addr
add wave /tb_top_1/compare_pass

echo "INFO: Signals added to wave window."

echo "INFO: Running simulation (run -all)..."
run -all
echo "INFO: Simulation finished. Waveform is now available in the wave window."