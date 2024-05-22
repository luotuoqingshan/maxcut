graphs = ["G$i" for i = 1:9] 
seed = 0
tol = 0.01

open(homedir()*"/maxcut/test_maxcut_manopt.txt", "w") do io
    for graph in graphs
        # remember to warmup each function
        println(io, "ulimit -d $((16 * 1024 * 1024)); "*
        "matlab -singleCompThread -batch \"cd ~/maxcut/;"*
        "test_maxcut(graph='G1', solver='manopt');"*
        "test_maxcut(graph='$graph', seed=$seed, tol=$tol, solver='manopt');\"")
    end
end
