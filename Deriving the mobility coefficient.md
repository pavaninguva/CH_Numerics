# Deriving various bits and pieces

## Diffusion Equation

We have
$$
\frac{\partial c}{\partial t} \frac{[mol]}{[m^{3}\cdot s]} = D \frac{[m^{2}]}{[s]}\frac{\partial^{2} c}{\partial x^{2}} \frac{[mol]}{[m^{5}]}
$$
When expressed in mole/volume fractions
$$
\frac{\partial \phi}{\partial t} \frac{[-]}{[s]} = D \frac{[m^{2}]}{[s]}\frac{\partial^{2}\phi}{\partial x^{2}} \frac{[-]}{[m^{2}]}
$$
We can also see what the units of our flux should be:
$$
\frac{\partial \phi}{\partial t} \frac{[-]}{[s]} = -\nabla \frac{[-]}{[m]} \cdot( -D\nabla \phi \frac{[m]}{[s]}) 
$$
So in terms of fractions, our flux has units of $m/s$. 

## Einstein's relationship

We start with Einstein's relationship
$$
D = M k_{B}T
$$
where $D$ is the diffusion coefficient, $M$ is the mobility, $k_{B}$ is Boltzmann's constant, and $T$ is the temperature. 

The mobility can be understood as the proportionality factor relating the particle's drift velocity to an applied force,
$$
v_{d} = MF
$$
The driving force in this case is gradients in the chemical potential, so we have $-\nabla \mu$ (note this arises from conservative forces $F = -\nabla U$)

Recall our units: 
$$
M \frac{[m^2]}{[J \cdot s]} = v_{d}\frac{[m]}{[s]} \frac{1}{F}\frac{[m]}{[J]}
$$
For our chemical potential driving force,
$$
-\nabla \frac{[1]}{[m]} \mu\frac{[J]}{[-]}
$$
Note that our $\mu$ is scaled by $k_{B}T$, so it is on a per molecule basis. If it were $J/mol$, then there would be factor of $\frac{1}{N_{A}}$ which corresponds to Avogadro's number.

Our flux can be written as the product of the concentration and velocity
$$
\mathbf{J}_{i} \frac{[m]}{[s]} = \phi_{i}\frac{[-]}{[-]}  v_{i}\frac{[m]}{[s]}
$$
Putting things together
$$
\mathbf{J}_{i} = -M\phi_{i}\nabla \mu
$$
So we see that the concentration dependence emerges as a consequence of the Einstein relation. Yay!! 

## Working with differences in chemical potential

So, we have
$$
\mathbf{J}_{i} = -\sum_{j=1}^{N} L_{ij}\nabla\mu_{j}
$$
Where $L_{ij}$ is the Onsanger coefficient. For the binary system
$$
\mathbf{J}_{1} = -L_{11}\nabla \mu_{1} - L_{12}\nabla \mu_{2}
$$
We have the following requirements for the coefficients

Onsanger reciprocal relations
$$
L_{ij} = L_{ji}
$$
and from mass conservation
$$
\sum_{i=1}^{N} \mathbf{J}_{i} = 0 \\
%
-L_{11} \nabla\mu_{1} -L_{12}\nabla\mu_{2} - L_{21}\nabla\mu_{1} - L_{22}\nabla\mu_{2} = 0 \\
%
-\nabla \mu_{1} (L_{11} + L_{21}) -\nabla \mu_{1}(L_{12} + L_{22}) = 0
$$
Since our gradients arent zero, we have 
$$
L_{11} + L_{21} = 0, \quad L_{12} + L_{22} = 0 \\
%
\Rightarrow \sum_{i}L_{ij} = 0
$$
Substituting this back, we get
$$
J_{1} = L_{12}\nabla(\mu_{1} - \mu_{2})
$$
More generally
$$
\mathbf{J}_{i} = \sum_{j} L_{ij} \nabla(\mu_{i} - \mu_{j})
$$

### Quick aside

We also have the following relationships which we can keep in our back pocket. 
$$
\mathcal{F} = \sum_{i}\phi_{i}\mu_{i} \Rightarrow \phi_{1}\mu_{1} + \phi_{2}\mu_{2} = \mathcal{F}
$$
And the Gibbs Duhem relationship:
$$
\sum_{i}\phi_{i} \nabla\mu_{i} = 0 \Rightarrow \phi_{1}\nabla\mu_{1} = -\phi_{2}\nabla\mu_{2}
$$

## Figuring out our expression for M / L

From Einstein's relation, we have
$$
L_{ij} = -\frac{\mathcal{D}_{ij}(\phi_{1},\phi_{2},\dots, \phi_{N})}{k_{B}T}\phi_{i}
$$
For a lattice fluid, we have
$$
\mathcal{D}_{ij} = D_{ij}\phi_{j},
$$
which arises from the modification of the diffusivity to account for the conditional probability that the molecule is able to diffuse on the lattice. 

## Figuring out the difference in chemical potentials

So we have the free energy per molecule / lattice site $F$
$$
\frac{F}{k_{B}T} = f(\phi_{1}) + \frac{\kappa}{2}(\nabla \phi_{1})^{2}
$$
By definition
$$
F = \phi_{1}\mu_{1} + \phi_{2}\mu_{2}
$$
To convert it into a free energy density, we multiply $F$ by $N_{0}$, which is the total number of molecules per unit volume. 
$$
\frac{N_{0}F}{k_{B}T} = (N_{1} + N_{2}) \left( f(\phi_{1}) + \frac{\kappa}{2}(\nabla \phi_{1})^{2} \right)
$$
Our chemical potential is given by 
$$
\mu_{i} = \frac{\delta N_{0}F}{\delta N_{i}} = \frac{\partial (N_{0}F)}{\partial N_{i}} - \nabla \cdot \frac{\partial (N_{0}F)}{\partial \nabla N_{i}}
$$
We also note that $\phi_{i} = \frac{N_{i}}{N_{0}}$. 

Computing the variation derivative term by term: 
$$
\frac{\partial (N_{0}F)}{\partial N_{1}} = F\frac{\partial N_{0}}{\partial N_{1}} + N_{0}\frac{\partial F}{\partial N_{1}}
$$

$$
\frac{\partial N_{0}}{\partial N_{1}} F= F
$$

$$
N_{0}\frac{\partial F}{\partial N_{1}} = N_{0}\frac{\partial F}{\partial \phi_{1}} \frac{\partial \phi_{1}}{\partial N_{1}} = N_{0}\frac{\partial f}{\partial \phi_{1}}\frac{N_{2}}{N_{0}^{2}} = \phi_{2}\frac{\partial f}{\partial \phi_{1}} = (1-\phi_{1})\frac{\partial f}{\partial \phi_{1}}
$$

So we get
$$
\frac{\partial N_{0}F}{\partial N_{1}} = F + (1-\phi_1)\frac{\partial f}{\partial \phi_1}
$$
For the second part of the derivative, i.e., involving $\nabla \phi_{1}$, we only have the term
$$
\frac{\partial (N_{0}F)}{\partial \nabla N_{1}} = F\frac{\partial N_{0}}{\partial \nabla N_{1}} + N_{0}\frac{\partial F}{\partial \nabla N_{1}}
$$
The first term is zero! For the second term, $N_{0}$ is outside the derivative. So we can proceed normally. We just need to worry about the gradient term,
$$
\frac{\kappa}{2}(\nabla\phi_{1})^{2}
$$

$$
\frac{\partial F}{\partial \nabla N_{1}} = \kappa(\nabla \phi)\frac{\partial \nabla\phi_{1}}{\partial \nabla N_{1}} = \kappa (\nabla \phi_{1}) \frac{N_{2}}{N_{0}^{2}}
$$

Taking the divergence
$$
\nabla \cdot (\kappa \frac{N_{2}}{N_{0}}\nabla\phi_{1}) = (1-\phi_{1})\kappa \nabla^{2}\phi_{1}
$$
Putting things together
$$
\mu_{1} = F + (1-\phi_{1})\frac{\partial f}{\partial \phi_{1}} - (1-\phi_{1})\kappa\nabla^{2}\phi_{1} \\
%
= F + (1-a)\left( \frac{\partial f}{\partial \phi_{1}} - \kappa \nabla^{2}\phi_{1} \right)
$$
For $\mu_{2}$, Let us try a tip to make life simpler. 

We know that
$$
\phi_{1} = \frac{N_{1}}{N_1 + N_2}, \quad \phi_{2} = \frac{N_{2}}{N_{1} + N_{2}}
$$
We have
$$
\mu_2 = \frac{\partial (N_0 F)}{\partial N_{2}} -\nabla \cdot  \frac{\partial (N_0 F)}{\partial (\nabla N_2)}
$$
Similarly
$$
\frac{\partial (N_0 F)}{\partial N_2} = N_0\frac{\partial F}{\partial N_{2}} + F\frac{\partial N_{0}}{\partial N_{2}} \\
%
= F + \frac{\partial F}{\partial \phi_1}\frac{\partial \phi_{1}}{\partial \phi_{2}} \frac{\partial \phi_{2}}{\partial N_2} \\
%
= F - \frac{\partial f}{\partial \phi_{1}}\frac{N_1}{N_0} \\
%
= F - \phi_{1}\frac{\partial f}{\partial \phi_1}
$$
For the second term

We first note that 
$$
\nabla \phi_{1} |_{N_{1}} = -\frac{N_{1}}{N_0^2}\nabla N_{2}
$$
The only term that is of relevance here is 
$$
\frac{\kappa}{2}(\nabla\phi_{1})^{2}
$$

$$
\frac{\partial N_{0}F}{\partial (\nabla N_{2})} = N_{0}\frac{\partial F}{\partial (\nabla N_{2})} \\
%
= N_{0}\frac{\kappa}{2}\frac{\partial}{\partial \nabla \phi_{1}} \left(  (\nabla\phi_{1})^{2}\right)\frac{\partial \nabla \phi_{1}}{\partial \nabla N_{2}} \\
%
= -\kappa\frac{N_{1}}{N_{0}}\nabla\phi_{1}
$$

Taking the divergence
$$
-\nabla \cdot \frac{\partial (N_{0}F)}{\partial (\nabla N_{2})} = \kappa \phi_{1} \nabla^{2}\phi_{1}
$$
Putting things together
$$
\mu_{2} = F - \phi_{1}\left(\frac{\partial f}{\partial \phi_{1}} - \kappa \nabla^{2}\phi_{1} \right)
$$


Trying Nitash's way (this feels a bit hand wavy tbh)

In the homogeneous case where $F = f(\phi_{1},\phi_{2})$ only, and working with the volume-based version

We can write the total derivative
$$
\frac{dF}{d\phi_{1}} = \frac{\partial F}{\partial \phi_{1}}\frac{d\phi_{1}}{d\phi_{1}} + \frac{\partial F}{\partial \phi_{2}}\frac{d\phi_{2}}{d\phi_{1}} \\
%
= \frac{\partial f}{\partial \phi_{1}} - \frac{\partial f}{\partial \phi_{2}} \\
%
= \mu_{1} - \mu_{2}
$$
We note that $\mu_{1} - \mu_{2}$ is equivalent to $\frac{df}{d\phi_{1}}$, when $f$ is expressed solely in terms of $\phi_1$. 

We also know that
$$
\mu_{1} = \frac{\partial f}{\partial \phi_1}, \quad \mu_{2} = \frac{\partial f}{\partial \phi_{2}}
$$

$$
f = \phi_{1}\mu_{1} + \phi_{2}\mu_{2}
$$

We thus have two equations with two unknowns
$$
f' = \mu_{1} - \mu_{2} \\
f = \phi_{1}\mu_{1} + \phi_{2}\mu_{2}
$$
Rearranging
$$
\mu_1 = f + (1-\phi_{1})f'
$$

$$
\mu_2 = f -\phi_{1}f'
$$



## Ternary System

We have 
$$
\frac{\mathcal{F}}{N_{0}k_{\text{B}T}} = \int_{V} f(\phi_{1}, \phi_{2}, \phi_{3}) + \frac{\kappa_{1}}{2}(\nabla \phi_{1})^{2} + \frac{\kappa_{2}}{2}(\nabla \phi_{2})^{2} + \kappa_{12}(\nabla \phi_{1})(\nabla \phi_{2}) \ dV
$$
Taking the variational derivative 
$$
\mu_{1} = \frac{\partial f}{\partial \phi_{1}} - \kappa_{1}\nabla^{2}\phi_{1} - \kappa_{12}\nabla^{2}\phi_{2}
$$

$$
\mu_{2} = \frac{\partial f}{\partial \phi_{2}} - \kappa_{2}\nabla^{2}\phi_{2} - \kappa_{12}\nabla^{2}\phi_{1}
$$

$$
\mu_{12} = \mu_{1} - \mu_{2} = \frac{\partial f}{\partial \phi_{1}} - \frac{\partial f}{\partial \phi_{2}} -(\kappa_{1}-\kappa_{12})\nabla^{2}\phi_{1} + (\kappa_{2} - \kappa_{12})\nabla^{2}\phi_{2}
$$

So our full ternary system is
$$
\begin{align}
\frac{\partial \phi_{1}}{\partial t} &= \nabla \cdot (D_{12}\phi_{1}\phi_{2}\nabla \mu_{12} + D_{13}\phi_{1}(1-\phi_{1}-\phi_{2})\nabla \mu_{13}), \\
     %
     \frac{\partial \phi_{2}}{\partial t} &= \nabla \cdot (-D_{12}\phi_{1}\phi_{2} \nabla\mu_{12} + D_{23}\phi_{2}(1-\phi_{1}-\phi_{2})\nabla \mu_{23}).
     \end{align}
$$

$$
\begin{align}
\mu_{12} &= \mu_{1} - \mu_{2} = \frac{\partial f}{\partial \phi_{1}} - \frac{\partial f}{\partial \phi_{2}} - (\kappa_{1} - \kappa_{12}) \nabla^{2} \phi_{1} + (\kappa_{2} - \kappa_{12})\nabla\phi_{2},  \\
%
\mu_{13} &= \mu_{1} - \mu_{3} = \frac{\partial f}{\partial \phi_{1}} - \frac{\partial f}{\partial \phi_{3}} - \kappa_{1}\nabla^{2}\phi_{1} - \kappa_{12}\nabla^{2}\phi_{2},  \\
%
\mu_{23} &= \mu_{2} - \mu_{3} = \frac{\partial f}{\partial \phi_{2}} - \frac{\partial f}{\partial \phi_{3}} -\kappa_{12}\nabla^{2}\phi_{1} -\kappa_{2}\nabla^{2}\phi_{2}.
\end{align}
$$

Introduce the scaling:
$$
t = \frac{R_{G,1}^{2}}{D_{12}}\tilde{t}, \quad \mathbf{x} = R_{G,1}\tilde{\mathbf{x}}
$$
We get
$$
\frac{\partial \phi_{1}}{\partial \tilde{t}} = \tilde{\nabla}\cdot \left( \phi_{1}\phi_{2}\tilde{\nabla}\tilde{\mu}_{12} + \frac{D_{13}}{D_{12}}\phi_{1}(1-\phi_{1}-\phi_{2}) \tilde{\nabla}\tilde{\mu}_{13} \right),\\
%
\frac{\partial \phi_{2}}{\partial \tilde{t}} = \tilde{\nabla}\cdot \left( -\phi_{1}\phi_{2} \tilde{\nabla}\tilde{\mu}_{12} + \frac{D_{23}}{D_{12}}\phi_{2}(1-\phi_{1}-\phi_{2}) \tilde{\nabla}\tilde{\mu}_{23}  \right),
$$

$$
\frac{\partial \tilde{f}}{\partial \phi_{1}} = \frac{\ln\phi_{1}}{N_{1}} + \frac{1}{N_{1}} + \chi_{12}\phi_{2} + \chi_{13}(1-\phi_{1} - \phi_{2}) \\
%
\frac{\partial \tilde{f}}{\partial \phi_{2}} = \frac{\ln{\phi_{2}}}{N_{2}} + \frac{1}{N_{2}} + \chi_{12}\phi_{1} + \chi_{23}(1-\phi_{1}-\phi_{2}) \\
%
\frac{\partial \tilde{f}}{\partial \phi_{3}} = \frac{\ln{(1-\phi_{1}-\phi_{2})}}{N_{3}} + \frac{1}{N_{3}} + \chi_{13}\phi_{1} + \chi_{23}\phi_{2}
$$

