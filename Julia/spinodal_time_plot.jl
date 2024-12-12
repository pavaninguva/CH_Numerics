using Plots
using CSV
using LaTeXStrings
using DataFrames


#Backward Euler 
df1 = CSV.read("spinodal_times_dt0.1_dx0.1.csv", DataFrame)
df2 = CSV.read("spinodal_times_dt0.1_dx0.05.csv",DataFrame)
df3 = CSV.read("spinodal_times_dt0.1_dx0.02.csv", DataFrame)
df4 = CSV.read("spinodal_times_dt0.05_dx0.1.csv",DataFrame)
df5 = CSV.read("spinodal_times_dt0.05_dx0.05.csv", DataFrame)
df6 = CSV.read("spinodal_times_dt0.05_dx0.02.csv",DataFrame)
df7 = CSV.read("spinodal_times_dt0.01_dx0.1.csv", DataFrame)
df8 = CSV.read("spinodal_times_dt0.01_dx0.05.csv",DataFrame)
df9 = CSV.read("spinodal_times_dt0.01_dx0.02.csv",DataFrame)


dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
labels = [
    L"\Delta t=0.1, \Delta x=0.1",
    L"\Delta t=0.1, \Delta x=0.05",
    L"\Delta t=0.1, \Delta x=0.02",
    L"\Delta t=0.05, \Delta x=0.1",
    L"\Delta t=0.05, \Delta x=0.05",
    L"\Delta t=0.05, \Delta x=0.02",
    L"\Delta t=0.01, \Delta x=0.1",
    L"\Delta t=0.01, \Delta x=0.05",
    L"\Delta t=0.01, \Delta x=0.02"
]

p1 = scatter(
    xlabel = L"\tau_0",
    ylabel = L"\tau",
    legend = :topleft,size=(500,500),tickfont=Plots.font("Computer Modern", 12), grid=false,
    legendfont=Plots.font("Computer Modern",8),dpi=300,xaxis=:log, yaxis=:log
)

colors = [:blue, :blue, :blue, :red, :red, :red, :green, :green, :green]
markers = [:circle, :square, :+, :circle, :square, :+,:circle, :square, :+ ]

for (i, df) in enumerate(dfs)
    scatter!(p1,
        df.tau0,
        df.tau,
        label = labels[i],
        color = colors[i],
        marker = markers[i],
        ms = 5,  # marker size
        alpha = 0.6
    )
end

#Generate line of best Fit
df_all = vcat(df1,df2,df3,df4,df5,df6,df7,df8,df9)

#Fit straight line y = mx
m1 = sum(df_all.tau0 .* df_all.tau) / sum(df_all.tau0.^2)
tau0_vals = range(minimum(df_all.tau0), maximum(df_all.tau0),100)
tau_vals = m1.*tau0_vals
plot!(p1,
    tau0_vals,
    tau_vals,
    label = "m = $(round(m1, digits=5))",
    color = :black,
    linewidth = 2,
    linestyle = :dash,
    alpha=0.5
)
savefig(p1,"./spinodal_times_impliciteuler1d.png")
display(p1)