# Max-Cut
This code repo is my modification to Manopt on max cut SDP. 
Please refer to the original repo for the excellent work from Boumal et. al.

# Our modification
I mainly want to stop the optimization by the suboptimality instead
of the gradient norm. I achieved it by exponentially decay the gradient norm until the desired suboptimality is achieved. I can also use the `stopfun` interface offered by `Manopt` to customize stopping criterion, but I found frequently checking the suboptimality a bit slow, so I chose to do it in this way. I believe they don't have significant difference.    

Also I provide a batch testing tool `gen_test_maxcut.jl`, take a look at this [README](https://github.com/luotuoqingshan/SketchyCGAL) for more info.

Basically you just install **ulimit** and **parallel** and run
```
cat test_maxcut_manopt.txt | parallel --jobs 9 --timeout 28800 {}
```