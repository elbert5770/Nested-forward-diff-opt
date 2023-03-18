using FiniteDiff
using ForwardDiff
using Optimization
using OptimizationOptimJL
using NonlinearSolve
using DifferentialEquations
using SteadyStateDiffEq
using Plots
using SciMLSensitivity

function sys!(du, u, (p,p_fix,kf,plateau,kd,start_time,end_time), t)
  
    du[1] = kf[1]*(1.0 - forcing_function(t,plateau,kd,start_time,end_time)) - p[1]*u[1]
    du[2] = p[1]*u[1] - p_fix[1]*u[2]
    du[3] = p_fix[1]*u[2] - p[2]*u[3]
   
    return nothing
end

function forcing_function(t,plateau,kd,start_time,end_time)
    if t < start_time
        return plateau*t
    else
        if t < end_time
            return plateau
        else
            if t > end_time
                return plateau*(exp(-kd*(t-end_time)))
            end
        end
    end
end



function loss_outer(probODE,prob_set_kf,probss,u03,Tmeas,Ymeas)
    function loss_inner(p_guess,(u0,p_fix,kf_guess,plateau,kd,start_time,end_time))
        # Optimize kf based on knowing u0[3] at steady state using current value of p_guess
        sol_kf = solve(remake(prob_set_kf,u0=eltype(p_guess).(kf_guess),p=(u0,p_guess,p_fix,plateau,kd,start_time,end_time)),BFGS())
        # Use optimized value of kf and current p_guess to calculate u0 at steady state (and also time=0)
        # Because system was assumed at steady state at time = 0
        sol_ss = solve(remake(probss,p=(p_guess,p_fix,sol_kf.u,0.0,kd,start_time,end_time)))
        # Solve ODE with current p_guess, u0 and optimized kf over timespan
        sol = solve(remake(probODE,u0=eltype(p_guess).(sol_ss.u),p=(p_guess,p_fix,sol_kf.u,plateau,kd,start_time,end_time)),dtmax=0.1)
        
        # Match ODE solution to 'measured' labeling
        Xpred = sol(Tmeas)
        Ypred = (u03.-Xpred[3,:])./u03
        return sum((Ypred.-Ymeas).^2)
    end
end


function set_kf_outer(probss,u03)
    function set_kf_inner(kf_guess,(u0,p_guess,p_fix,plateau,kd,start_time,end_time))
        # Solve for steady state with current p_guess and an initial guess of kf and u0
        sol = solve(remake(probss,u0=eltype(kf_guess).(u0),p=(p_guess,p_fix,kf_guess,plateau,kd,start_time,end_time)))
        # Compare kf at steady state to measured u0[3]
        return (sol[3].-u03).^2
    end
end

function main()
    #### Parameters and state variables
    kf0= [10.0] #true value of zero order production rate
    p0 = [1.0,0.25,0.5] #true values of parameters
    p_fix = [p0[2]] # a parameter that is known, note that the system is otherwise unidentifiable
    p_var = [p0[1],p0[3]] # Unknown parameters
    u0_guess = [1000.0,750.0,20.0] #initial guess at steady state of state variables, except u0[3] is known to be 20.0
    u03 = 20.0 # measured value
    plateau = 0.0 # Zero order production rate changes at time = 0, reaches a fractional change given by plateau at start_time
    kd = 1.0 # Rate constant for exponential decay back to original value starting at end_time
    start_time = 1.0
    end_time = 10.0

    #### Set up ODE problem
    t_end = 36.0 # Length of simulation
    tspan = (0.0,t_end) # timespan of simulation
    probODE = ODEProblem(sys!,u0_guess,tspan,(p_var,p_fix,kf0,plateau,kd,start_time,end_time))
    
    #### Set up steady state problem
    probss = SteadyStateProblem(probODE)
    sol_ss = solve(probss)
    u0_ss = sol_ss.u
    @show "u0_ss true",sol_ss.u
    
    #### Solve dynamic system for true solution with plateau 0.5
    plateau = 0.5
    
    sol_true = solve(remake(probODE,u0=u0_ss,p=(p_var,p_fix,kf0,plateau,kd,start_time,end_time)),dtmax=0.1)
    #display(plot(sol_true[3,:]))
    
    #### Hourly solution to simulate measurements
    Tmeas = collect(0:t_end) # Timepoints
    Xmeas = sol_true(Tmeas) # Solution at timepoints
    #@show Xmeas
    Ymeas = (u03.-Xmeas[3,:])./u03 # Measured quantity is given by the shown transformation
    #@show Ymeas

    # Plot true solution with simulated measurements
    # p1 = plot(sol_true.t,(u03.-sol_true[3,:])./u03,label="solution with true values",title="True solution")
    # scatter!(p1,Tmeas,Ymeas,label="'measured' quantity")
    # display(plot(p1))
    
    #### Reset plateau to zero to solve for steady state with p_guess and kf_guess (initial values)
    p_guess = [10.0,3.0] #inital guess at values in p_val
    kf_guess = [100.0] #initial guess at kf0
    

    #### Set up optimization problem to determine kf for current value of p_guess
    set_kf = set_kf_outer(probss,u03)
    optkf = OptimizationFunction(set_kf,Optimization.AutoForwardDiff())
    prob_set_kf = OptimizationProblem(optkf,kf_guess,(u0_guess,p_guess,p_fix,plateau,kd,start_time,end_time))
    
    
    #### Set up optimization problem for p_guess
    loss = loss_outer(probODE,prob_set_kf,probss,u03,Tmeas,Ymeas)
    
    # Compare gradients by finite diff and forward diff
    @show FiniteDiff.finite_difference_gradient(x -> loss(x,(u0_guess,p_fix,kf_guess,plateau,kd,start_time,end_time)), p_guess)
    @show ForwardDiff.gradient(x -> loss(x,(u0_guess,p_fix,kf_guess,plateau,kd,start_time,end_time)), p_guess)
    @show "Finite diff gradient is similar to but not equal to Forward diff gradient"

    #### Optimization problem with derivative
    optp = OptimizationFunction(loss,Optimization.AutoForwardDiff())
    prob_find_p = OptimizationProblem(optp,p_guess,(u0_guess,p_fix,kf_guess,plateau,kd,start_time,end_time))
    sol_p = solve(prob_find_p,Optim.NewtonTrustRegion())
    # Optional additional optimizations
    # for i in 1:2
    #     sol_p = solve(remake(prob_find_p,u0=sol_p.u),Optim.BFGS())
    # end
    # sol_p = solve(remake(prob_find_p,u0=sol_p.u),Optim.NewtonTrustRegion())

    #### Optimization problem without derivative
    # prob_find_p = OptimizationProblem(loss,p_guess,(u0_guess,p_fix,kf_guess,plateau,kd,start_time,end_time))  
    # sol_p = solve(prob_find_p,NelderMead(),allow_f_increases=true)
    
    @show "Optimized p",sol_p.u
    @show sol_p.original


    #### Analyze results with optimized p_guess
    sol_kf = solve(remake(prob_set_kf,u0=kf_guess,p=(u0_guess,sol_p.u,p_fix,plateau,kd,start_time,end_time)),BFGS())
    @show "Final answer for kf:",sol_kf.u,kf0
    sol_ss = solve(remake(probss,p=(sol_p.u,p_fix,sol_kf.u,plateau,kd,start_time,end_time)))
    @show "Final answer for u_ss:",sol_ss.u,u0_ss
    # Solve ODE over time with final value for u0, kf and p 
    sol = solve(remake(probODE,u0=sol_ss.u,p=(sol_p.u,p_fix,sol_kf.u,plateau,kd,start_time,end_time)),dtmax=0.1)
    p1 = plot(sol.t,(u03.-sol[3,:])./u03,label="p_guess",title="Optimized solution")
    scatter!(p1,Tmeas,Ymeas,label="measured labeling")
    xlabel!(p1,"time (h)")
    ylabel!(p1,"fraction labeled")
    display(p1)


    #### Compare results above to solution with true p
    sol_kf = solve(remake(prob_set_kf,u0=kf_guess,p=(u0_guess,p_var,p_fix,plateau,kd,start_time,end_time)),BFGS())
     @show "Solve for kf when p_var is correct",sol_kf.u,kf0
    sol_ss = solve(remake(probss,p=(p_var,p_fix,sol_kf.u,plateau,kd,start_time,end_time)))
    @show "Solve for u_ss when p_var is correct",sol_ss.u,u0_ss
    # Solve ODE over time with correct values for u0, kf and p
    sol = solve(remake(probODE,u0=sol_ss.u,p=(p_var,p_fix,sol_kf.u,plateau,kd,start_time,end_time)),dtmax=0.1)
    p1 = plot(sol.t,(u03.-sol[3,:])./u03,label="p_guess",title="True solution")
    scatter!(p1,Tmeas,Ymeas,label="measured labeling")
    xlabel!(p1,"time (h)")
    ylabel!(p1,"fraction labeled")
    display(p1)
    Xpred = sol(Tmeas)
    Ypred = (u03.-Xpred[3,:])./u03
    @show "Error in ODE solution with true values",sum((Ypred.-Ymeas).^2)

    return nothing
end

main()