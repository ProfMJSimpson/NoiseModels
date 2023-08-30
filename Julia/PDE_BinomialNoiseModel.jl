using Plots, DifferentialEquations, Interpolations, Distributions, NLopt, SimpleDiffEq, Dierckx
gr()


function diff!(du,u,p,t)
dx,N,D,λ,β=p 
for i in 2:N-1
du[i]=tanh(β*t)*(D*(u[i-1]-2*u[i]+u[i+1])/dx^2+λ*u[i]*(1-u[i])) 
end
du[1]=tanh(β*t)*(D*(u[2]-u[1])/dx^2+λ*u[1]*(1-u[1])) 
du[N]=tanh(β*t)*(D*(u[N-1]-u[N])/dx^2+λ*u[N]*(1-u[N])) 
end
  
    


function pdesolver(L,dx,N,T,ic,D,λ,β)
p=(dx,N,D,λ,β)
tspan=(0.0,maximum(T))
prob=ODEProblem(diff!,ic,tspan,p)
#alg=Tsit5()
alg=Heun()
sol=solve(prob,alg,saveat=T);
return sol
end



function Optimise(fun,θ₀,lb,ub)
    
    tomax=(θ,∂θ)->fun(θ)
    opt=Opt(:LN_BOBYQA,length(θ₀))
    opt.max_objective=tomax
    opt.lower_bounds=lb      
    opt.upper_bounds=ub
    opt.maxtime=1*10
    res = optimize(opt,θ₀)
    return res[[2,1]]

end




L=1900;
T=[0,12,24,36,48];
dx=5.0;
N=Int(round(L/dx))+1;
D=1400;
λ=0.09;
β = 0.1;


numsol=zeros(N,length(T));
ic=zeros(N);
data=zeros(38,length(T));
count=zeros(38,length(T));

xlocdata =[25,75,125,175,225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125,1175,1225,1275,1325,1375,1425,1475,1525,1575,1625,1675,1725,1775,1825,1875];
count[:,1]=[89  78  87  79  79  79  84  82  85  74  71  43  6  3  2  1  2  1  2  2  3  1  2  2  1  1  2  5  41  82  79  77  94  76  79  79  88  74];
count[:,2]=[95  80  94  89  87  90  80  79  77  71  75  62  36  8  2  0  1  2  1  1  1  2  2  2  1  0  9  34  62  82  79  75  88  89  84  89  87  89];
count[:,3]=[120  112  112  109  98  102  100  101  94  92  84  79  61  64  51  37  21  15  9  4  2  5  16  38  45  50  60  71  80  82  97  94  92  105  103  98  104  109];
count[:,4]=[127  116  117  117  114  114  112  103  97  104  92  87  89  67  72  67  71  65  60  61  67  64  73  71  79  75  85  82  89  87  97  107  104  106  118  116  111  119];
count[:,5]=[147  122  123  120  125  119  119  112  109  107  110  102  97  99  94  94  93  101  93  94  86  101  94  91  98  99  97  96  103  107  106  104  119  114  124  122  123  120];
count=round.(Int,count)

#data[:,1]=[1.25E-03	1.09E-03	1.21E-03	1.11E-03	1.11E-03	1.10E-03	1.17E-03	1.15E-03	1.19E-03	1.03E-03	9.88E-04	6.06E-04	7.93E-05	3.73E-05	2.80E-05	1.86E-05	2.80E-05	1.86E-05	2.33E-05	2.33E-05	4.66E-05	1.40E-05	2.33E-05	2.80E-05	1.86E-05	9.32E-06	2.33E-05	6.99E-05	5.73E-04	1.15E-03	1.10E-03	1.07E-03	1.32E-03	1.06E-03	1.10E-03	1.11E-03	1.23E-03	1.04E-03]/1.7E-3;
#data[:,2]=[1.33E-03	1.12E-03	1.31E-03	1.24E-03	1.22E-03	1.26E-03	1.12E-03	1.11E-03	1.07E-03	9.93E-04	1.05E-03	8.72E-04	5.08E-04	1.17E-04	2.33E-05	0.00E+00	1.40E-05	2.80E-05	1.86E-05	9.32E-06	9.32E-06	2.33E-05	2.80E-05	2.80E-05	9.32E-06	4.66E-06	1.21E-04	4.71E-04	8.72E-04	1.15E-03	1.10E-03	1.05E-03	1.23E-03	1.24E-03	1.17E-03	1.25E-03	1.22E-03	1.25E-03]/1.7E-3; 
#data[:,3]=[1.68E-03	1.56E-03	1.57E-03	1.53E-03	1.37E-03	1.42E-03	1.40E-03	1.41E-03	1.31E-03	1.28E-03	1.17E-03	1.11E-03	8.58E-04	8.90E-04	7.09E-04	5.17E-04	2.89E-04	2.14E-04	1.21E-04	5.59E-05	2.33E-05	7.46E-05	2.19E-04	5.31E-04	6.34E-04	6.99E-04	8.44E-04	9.93E-04	1.12E-03	1.15E-03	1.36E-03	1.31E-03	1.29E-03	1.47E-03	1.44E-03	1.37E-03	1.45E-03	1.52E-03]/1.7E-3;
#data[:,4]=[1.78E-03	1.62E-03	1.63E-03	1.64E-03	1.59E-03	1.59E-03	1.56E-03	1.44E-03	1.35E-03	1.45E-03	1.28E-03	1.22E-03	1.25E-03	9.37E-04	1.01E-03	9.42E-04	9.93E-04	9.04E-04	8.44E-04	8.48E-04	9.32E-04	9.00E-04	1.02E-03	9.88E-04	1.10E-03	1.05E-03	1.19E-03	1.14E-03	1.25E-03	1.22E-03	1.36E-03	1.49E-03	1.46E-03	1.48E-03	1.65E-03	1.62E-03	1.55E-03	1.67E-03]/1.7E-3;
#data[:,5]=[2.06E-03	1.71E-03	1.72E-03	1.68E-03	1.75E-03	1.67E-03	1.66E-03	1.56E-03	1.52E-03	1.49E-03	1.54E-03	1.42E-03	1.36E-03	1.38E-03	1.32E-03	1.32E-03	1.30E-03	1.41E-03	1.30E-03	1.31E-03	1.20E-03	1.41E-03	1.31E-03	1.27E-03	1.37E-03	1.38E-03	1.35E-03	1.34E-03	1.44E-03	1.49E-03	1.48E-03	1.46E-03	1.67E-03	1.59E-03	1.73E-03	1.71E-03	1.72E-03	1.68E-03]/1.7E-3;

count = data*1.7e-3*50*1430;
data=count/(1.7e-3*50*1430);
count=round.(Int,count)
#Linear Interpolate the initial data onto the finite difference mesh
interpr = linear_interpolation(xlocdata,data[:,1]);
interpr2 = linear_interpolation(xlocdata,data[:,2]);
interpr3 = linear_interpolation(xlocdata,data[:,3]);
interpr4 = linear_interpolation(xlocdata,data[:,4]);
interpr5 = linear_interpolation(xlocdata,data[:,5]);


for i in 1:N
xx = (i-1)*dx
    if xx >= 25 && xx <= 1875
    ic[i] = interpr(xx)
    elseif xx < 25
    ic[i] = data[1,1]
    elseif xx > 1875
    ic[i] = data[end,1]
    end
end



@time numsol=pdesolver(L,dx,N,T,ic,D,λ,β); 


 p1 = scatter(xlocdata,data[:,1],legend=false,msw=0,ms=3,color=:blue,msa=:blue,ylims=(0,1.2))
 p1 = plot!(0:dx:L,numsol[:,1],lw=2,legend=false,color=:blue,ylims=(0,1.2))
 p2 = scatter(xlocdata,data[:,2],legend=false,msw=0,ms=3,color=:blue,msa=:blue,ylims=(0,1.2))
 p2 = plot!(0:dx:L,numsol[:,2],lw=2,legend=false,color=:blue,ylims=(0,1.2))
 p3 = scatter(xlocdata,data[:,3],legend=false,msw=0,ms=3,color=:blue,msa=:blue,ylims=(0,1.2))
 p3 = plot!(0:dx:L,numsol[:,3],lw=2,legend=false,color=:blue,ylims=(0,1.2))
 p4 = scatter(xlocdata,data[:,4],legend=false,msw=0,ms=3,color=:blue,msa=:blue,ylims=(0,1.2))
 p4 = plot!(0:dx:L,numsol[:,4],lw=2,legend=false,color=:blue,ylims=(0,1.2))
 p5 = scatter(xlocdata,data[:,5],legend=false,msw=0,ms=3,color=:blue,msa=:blue,ylims=(0,1.2))
 p5 = plot!(0:dx:L,numsol[:,5],lw=2,legend=false,color=:blue,ylims=(0,1.2))
 p6=plot(p1,p2,p3,p4,p5,layout=(5,1))


function bloglhood(count,ic,a)
numsol=pdesolver(L,dx,N,T,ic,a[1],a[2],a[3]) 
u2 = linear_interpolation(0:dx:L,numsol[:,2]);
u3 = linear_interpolation(0:dx:L,numsol[:,3]);
u4 = linear_interpolation(0:dx:L,numsol[:,4]);
u5 = linear_interpolation(0:dx:L,numsol[:,5]);
e=0
for ii in 1:size(count)[1]
e+=(+log(u2(xlocdata[ii])^count[ii,2]*(1-u2(xlocdata[ii]))^(122-count[ii,2]))
    +log(u3(xlocdata[ii])^count[ii,3]*(1-u3(xlocdata[ii]))^(122-count[ii,3]))
    +log(u4(xlocdata[ii])^count[ii,4]*(1-u4(xlocdata[ii]))^(122-count[ii,4]))
    +log(u5(xlocdata[ii])^count[ii,5]*(1-u5(xlocdata[ii]))^(122-count[ii,5]))
   )   
end
    return e
end












a=zeros(3)
function bfunmle(a)
return bloglhood(count,ic,a)
end


θG=[1000,0.06,0.1]
lb=[500, 0.01,0.01]
ub=[2000,0.10,1]
@time (xopt,fopt)=Optimise(bfunmle,θG,lb,ub)
fmle=fopt
Dmle=xopt[1]; 
λmle=xopt[2];
βmle=xopt[3]; 

@time numsolb=pdesolver(L,dx,N,T,ic,Dmle,λmle,βmle); 
p1 = scatter(xlocdata,data[:,1],legend=false,msw=0,ms=5,color=:blue,msa=:blue,ylims=(0.0,1.1))
p1 = plot!(0:dx:L,numsolb[:,1],lw=2,legend=false,color=:blue,ylims=(0.0,1.1))
p1 = scatter!(xlocdata,data[:,2],legend=false,msw=0,ms=5,color=:red,msa=:red,ylims=(-0.0,1.1))
p1 = plot!(0:dx:L,numsolb[:,2],lw=2,legend=false,color=:red,ylims=(-0.0,1.1))
p1 = scatter!(xlocdata,data[:,3],legend=false,msw=0,ms=5,color=:green,msa=:green,ylims=(-0.0,1.1))
p1 = plot!(0:dx:L,numsolb[:,3],lw=2,legend=false,color=:green,ylims=(-0.0,1.1))
p1 = scatter!(xlocdata,data[:,4],legend=false,msw=0,ms=5,color=:coral,msa=:coral,ylims=(-0.0,1.1))
p1 = plot!(0:dx:L,numsolb[:,4],lw=2,legend=false,color=:coral,ylims=(-0.0,1.1))
p1 = scatter!(xlocdata,data[:,5],legend=false,msw=0,ms=5,color=:orange,msa=:orange,ylims=(-0.0,1.1))
p1 = plot!(0:dx:L,numsolb[:,5],lw=2,legend=false,color=:orange,ylims=(-0.0,1.1),xlabel="x",ylabel="u(x,t)/K")
savefig(p1,"MLE_Binomial.pdf")


Dmin=1200;
Dmax=1650;
λmin=0.090;
λmax=0.109;
βmin=0.045;
βmax=0.072;



df=3
llstar=-quantile(Chisq(df),0.95)/2
M=1000
Dsampled=zeros(M)
λsampled=zeros(M)
βsampled=zeros(M)
lls=zeros(M)
kount = 0

while kount < M
Dg=rand(Uniform(Dmin,Dmax))
λg=rand(Uniform(λmin,λmax))
βg=rand(Uniform(βmin,βmax))
    if (bloglhood(count,ic,[Dg,λg,βg])-fmle) >= llstar
    kount+=1
    println(kount)
    lls[kount]=bloglhood(count,ic,[Dg,λg,βg])-fmle
    Dsampled[kount]=Dg;
    λsampled[kount]=λg;
    βsampled[kount]=βg;
    end
end

q1=scatter(Dsampled,legend=false)
q1=hline!([Dmin,Dmax],legend=false)

q2=scatter(λsampled,legend=false)
q2=hline!([λmin,λmax],legend=false)

q3=scatter(βsampled,legend=false)
q3=hline!([βmin,βmax],legend=false)






xx = 0:dx:L
lower2=10*ones(length(xx))
lower3=10*ones(length(xx))
lower4=10*ones(length(xx))
lower5=10*ones(length(xx))
upper2=zeros(length(xx))
upper3=zeros(length(xx))
upper4=zeros(length(xx))
upper5=zeros(length(xx))

for i in 1:M
println(i)
numsol=pdesolver(L,dx,N,T,ic,Dsampled[i],λsampled[i],βsampled[i]);

p2 = linear_interpolation(0:dx:L,numsol[:,2]);
p3 = linear_interpolation(0:dx:L,numsol[:,3]);
p4 = linear_interpolation(0:dx:L,numsol[:,4]);
p5 = linear_interpolation(0:dx:L,numsol[:,5]);

for j in 1:length(xx)
    if p2(xx[j]) < lower2[j] 
        lower2[j] = p2(xx[j])
    end
    
    if p3(xx[j]) < lower3[j] 
        lower3[j] = p3(xx[j])
    end

    if p4(xx[j]) < lower4[j] 
        lower4[j] = p4(xx[j])
    end
    
    if p5(xx[j]) < lower5[j] 
        lower5[j] = p5(xx[j])
    end

    if p2(xx[j]) > upper2[j] 
        upper2[j] = p2(xx[j])
    end
    
    if p3(xx[j]) > upper3[j] 
        upper3[j] = p3(xx[j])
    end

    if p4(xx[j]) > upper4[j] 
        upper4[j] = p4(xx[j])
    end
    
    if p5(xx[j]) > upper5[j] 
        upper5[j] = p5(xx[j])
    end

end



end


numsol=pdesolver(L,dx,N,T,ic,Dmle,λmle,βmle); 
qq1=plot(xx,lower2,lw=0,fillrange=upper2,fillalpha=0.40,color=:red,label=false,xlims=(0,L),ylims=(0,1.1))
qq1=plot!(0:dx:L,numsol[:,2],lw=4,color=:red,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(0,1.1))
qq2=plot(xx,lower3,lw=0,fillrange=upper3,fillalpha=0.40,color=:green,label=false,xlims=(0,L),ylims=(0,1.1))
qq2=plot!(0:dx:L,numsol[:,3],lw=4,color=:green,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(0,1.1))
qq3=plot(xx,lower4,lw=0,fillrange=upper4,fillalpha=0.40,color=:coral,label=false,xlims=(0,L),ylims=(0,1.1))
qq3=plot!(0:dx:L,numsol[:,4],lw=4,color=:coral,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(0,1.1))
qq4=plot(xx,lower5,lw=0,fillrange=upper5,fillalpha=0.40,color=:orange,label=false,xlims=(0,L),ylims=(0,1.1))
qq4=plot!(0:dx:L,numsol[:,5],lw=4,color=:orange,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(0,1.1))
qq5=plot(qq1,qq2,qq3,qq4,layout=(2,2))
savefig(qq5, "BinomialMeanPredictions.pdf")





xx = 0:dx:L
lower2=10*ones(length(xx))
lower3=10*ones(length(xx))
lower4=10*ones(length(xx))
lower5=10*ones(length(xx))
upper2=zeros(length(xx))
upper3=zeros(length(xx))
upper4=zeros(length(xx))
upper5=zeros(length(xx))



for i in 1:M
println(i)
numsol=pdesolver(L,dx,N,T,ic,Dsampled[i],λsampled[i],βsampled[i]);

p2 = linear_interpolation(0:dx:L,numsol[:,2]);
p3 = linear_interpolation(0:dx:L,numsol[:,3]);
p4 = linear_interpolation(0:dx:L,numsol[:,4]);
p5 = linear_interpolation(0:dx:L,numsol[:,5]);

    for j in 1:length(xx)

        if (quantile(Binomial(122,p2(xx[j])),[.05,.95])[1])/122 < lower2[j] 
            lower2[j] = (quantile(Binomial(122,p2(xx[j])),[.05,.95])[1])/122
        end
       
        if (quantile(Binomial(122,p3(xx[j])),[.05,.95])[1])/122 < lower3[j] 
            lower3[j] = (quantile(Binomial(122,p3(xx[j])),[.05,.95])[1])/122
        end

        if (quantile(Binomial(122,p4(xx[j])),[.05,.95])[1])/122 < lower4[j] 
            lower4[j] = (quantile(Binomial(122,p4(xx[j])),[.05,.95])[1])/122
        end

        if (quantile(Binomial(122,p5(xx[j])),[.05,.95])[1])/122 < lower5[j] 
            lower5[j] = (quantile(Binomial(122,p5(xx[j])),[.05,.95])[1])/122
        end

        if (quantile(Binomial(122,p2(xx[j])),[.05,.95])[2])/122 > upper2[j] 
            upper2[j] = (quantile(Binomial(122,p2(xx[j])),[.05,.95])[2])/122
        end
       
        if (quantile(Binomial(122,p3(xx[j])),[.05,.95])[2])/122 > upper3[j] 
            upper3[j] = (quantile(Binomial(122,p3(xx[j])),[.05,.95])[2])/122
        end

        if (quantile(Binomial(122,p4(xx[j])),[.05,.95])[2])/122 > upper4[j] 
            upper4[j] = (quantile(Binomial(122,p4(xx[j])),[.05,.95])[2])/122
        end

        if (quantile(Binomial(122,p5(xx[j])),[.05,.95])[2])/122 > upper5[j] 
            upper5[j] = (quantile(Binomial(122,p5(xx[j])),[.05,.95])[2])/122
        end





    end



end


numsol=pdesolver(L,dx,N,T,ic,Dmle,λmle,βmle); 
qq1=plot(xx,lower2,lw=0,fillrange=upper2,fillalpha=0.40,color=:red,label=false,xlims=(0,L),ylims=(-0.1,1.1))
qq1=plot!(0:dx:L,numsol[:,2],lw=4,color=:red,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(-0.1,1.1))
qq2=plot(xx,lower3,lw=0,fillrange=upper3,fillalpha=0.40,color=:green,label=false,xlims=(0,L),ylims=(-0.1,1.1))
qq2=plot!(0:dx:L,numsol[:,3],lw=4,color=:green,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(-0.1,1.1))
qq3=plot(xx,lower4,lw=0,fillrange=upper4,fillalpha=0.40,color=:coral,label=false,xlims=(0,L),ylims=(-0.1,1.1))
qq3=plot!(0:dx:L,numsol[:,4],lw=4,color=:coral,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(-0.1,1.1))
qq4=plot(xx,lower5,lw=0,fillrange=upper5,fillalpha=0.40,color=:orange,label=false,xlims=(0,L),ylims=(-0.1,1.1))
qq4=plot!(0:dx:L,numsol[:,5],lw=4,color=:orange,ls=:dash,label=false,xlims=(0,L),xlabel="x",ylabel="u(x,t)/K",ylims=(-0.1,1.1))
qq5=plot(qq1,qq2,qq3,qq4,layout=(2,2))
savefig(qq5, "BinomialMDataPredictions.pdf")
















df=1
llstar=-quantile(Chisq(df),0.95)/2



function univariateD(D)
    a=zeros(2)    
    function funD(a)
    return bloglhood(count,ic,[D,a[1],a[2]])
    end
    θG=[λmle,βmle]
    lb=[0.01,0.001] 
    ub=[0.50,1.0] 
    (xopt,fopt)=Optimise(funD,θG,lb,ub)
    return fopt,xopt
    end 
    fb(x) = univariateD(x)[1]


#Take a grid of M points to plot the univariate profile likelihood
M=30;
Drange=LinRange(1100,1900,M)
ff=zeros(M)
for i in 1:M
    ff[i]=univariateD(Drange[i])[1]
    println(i)
end

q1=scatter(Drange,ff.-maximum(ff),ylims=(-3,0),legend=false,lw=3,color=:blue)
q1=hline!([llstar],legend=false,lw=3)
q1=vline!([Dmle],legend=false,xlabel="D",ylabel="ll",lw=3,color=:blue)
spl=Spline1D(Drange,ff.-maximum(ff),w=ones(length(Drange)),k=3,bc="nearest",s=0.01)
yy=evaluate(spl,Drange)
q1=plot!(Drange,yy,lw=2,lc=:blue,ylims=(-3,0),xlims=(Drange[1],Drange[end]))
savefig(q1, "D_profile_binomial.pdf")


function univariateλ(λ)
a=zeros(2)    
function funλ(a)
return bloglhood(count,ic,[a[1],λ,a[2]])
end
θG=[Dmle,βmle]
lb=[500,0.01] 
ub=[3000,1.0] 
(xopt,fopt)=Optimise(funλ,θG,lb,ub)
return fopt,xopt
end 
f(x) = univariateλ(x)[1]



    #Take a grid of M points to plot the univariate profile likelihood

λrange=LinRange(0.07,0.11,M)
ff=zeros(M)

for i in 1:M
ff[i]=univariateλ(λrange[i])[1]       
println(i)
end
    
q1=scatter(λrange,ff.-maximum(ff),ylims=(-3,0.0),legend=false,lw=3,lc=:blue)
q1=hline!([llstar],legend=false,lw=3)
q1=vline!([λmle],legend=false,xlabel="λ",ylabel="ll",lw=3,lc=:blue)
spl=Spline1D(λrange,ff.-maximum(ff),w=ones(length(λrange)),k=3,bc="nearest",s=4.)
    yy=evaluate(spl,λrange)
    q1=plot!(λrange,yy,lw=2,lc=:blue,ylims=(-3,0),xlims=(λrange[1],λrange[end]))
savefig(q1, "λ_profile_binomial.pdf")
    

function univariateβ(β)
a=zeros(2)    
function funβ(a)
return bloglhood(count,ic,[a[1],a[2],β])
end
θG=[Dmle,λmle]
lb=[500,0.01] 
ub=[3000,0.5] 
(xopt,fopt)=Optimise(funβ,θG,lb,ub)
return fopt,xopt
end 
f(x) = univariateβ(x)[1]

#Take a grid of M points to plot the univariate profile likelihood
βrange=LinRange(0.02,0.1,M)
ff=zeros(M)
for i in 1:M
ff[i]=univariateβ(βrange[i])[1]        
println(i)
end
    
q1=scatter(βrange,ff.-maximum(ff),ylims=(-3,0.0),legend=false,lw=3,lc=:blue)
q1=hline!([llstar],legend=false,lw=3)
q1=vline!([βmle],legend=false,xlabel="β",ylabel="ll",lw=3,lc=:blue)
spl=Spline1D(βrange,ff.-maximum(ff),w=ones(length(βrange)),k=3,bc="nearest",s=1.0)
yy=evaluate(spl,βrange)
q1=plot!(βrange,yy,lw=2,lc=:blue,ylims=(-3,0),xlims=(βrange[1],βrange[end]))
savefig(q1, "β_profile_binomial.pdf")


        