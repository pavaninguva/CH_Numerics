# Deriving $\kappa$ for FH

## Binary

Our FH expression is
$$
g(\phi_{1},\phi_{2}) =\frac{\phi_{1}}{N_{1}}\ln(\phi_{1}) + \frac{\phi_{2}}{N_{2}}\ln(\phi_{2}) + \chi\phi_{1}\phi_{2}
$$
We have our Free energy functional which looks like: 
$$
\mathcal{G} = \int_{V} g(\phi_{1},\phi_{2}) + \frac{\kappa}{2}(\nabla \phi_{1})^{2} \ dV
$$
We can split $g$ and $\kappa$ into Entropic and Enthalpic contributions i.e., $g = g_{ideal} + g_{residual}$ and $\kappa = \kappa_{entropic} + \kappa_{enthalpic}$. 

To derive $\kappa_{entropic}$ first, we introduce a perturbation of the form
$$
\bar{\phi}_{i} = \phi_{i} - \epsilon_{i}, \quad \epsilon_{i} = \frac{R_{G,i}^{2}}{6}\nabla^{2}\phi_{i}.
$$
We need some basic algebraic identities to make progress on the expansion. 

#### Helpful aside 1: Expansion of $\ln{(x-\epsilon)}$

Consider $\ln{(x-\epsilon)}$, we can rewrite it as
$$
\ln{(x-\epsilon)} = \ln{\left(x(1-\frac{\epsilon}{x})\right)} \\
= \ln{(x)} + \ln{\left(1-\frac{\epsilon}{x}\right)}
$$
We can use a Taylor series expansion for the second term by introducing a transformation $u = \frac{\epsilon}{x}$, which neatly gives us the expansion
$$
\ln{\left(1-u\right)} = -u - \frac{u^{2}}{2} - \frac{u^{3}}{3} + \dots
$$
We are only interested in the first order terms. So we get
$$
\ln{(x-\epsilon)} = \ln{(x)} - \frac{\epsilon}{x} + \dots
$$
Back to things! 

Starting with
$$
\int_{V} \frac{\phi_{1} - \epsilon_{1}}{N_{1}}\ln{(\phi_{1}-\epsilon_{1})} + \frac{\phi_{2} - \epsilon_{2}}{N_{2}}\ln{(\phi_{2} - \epsilon_{2})} \ dV \\
%
$$
And inserting our expansion result
$$
\int_{V} \frac{1}{N_{1}}\left( (\phi_{1}-\epsilon_{1})\left( \ln{\phi_{1}} - \frac{\epsilon_{1}}{\phi_{1}} \right) \right) + \frac{1}{N_{2}}\left( (\phi_{2}-\epsilon_{2})\left( \ln{\phi_{2}} - \frac{\epsilon_{2}}{\phi_{2}} \right) \right) \ dV
$$
Expanding out more and retaining terms of $\mathcal{O}(\epsilon)$, 
$$
\int_{V} \frac{1}{N_{1}} \left(\phi_{1}\ln{\phi_{1}} -\epsilon_{1} \ln{\phi_{1}} - \epsilon_{1} \right) + \frac{1}{N_{2}}\left(\phi_{2} \ln{\phi_{2}} - \epsilon_{2}\ln{\phi_{2}} -\epsilon_{2}\right) \ dV
$$
Splitting up 
$$
\mathcal{O}(1): \\
\int_{V} \frac{\phi_{1}}{N_{1}}\ln{\phi_{1}} + \frac{\phi_{2}}{N_{2}} \ln{\phi_{2}} \ dV \\
%
\mathcal{O}(\epsilon): \\
\int_{V} -\frac{1}{N_{1}}\epsilon_{1}\ln{\phi_{1}} - \frac{\epsilon_{1}}{N_{1}}  - \frac{1}{N_{2}}\epsilon_{2} \ln{\phi_{2}} - \frac{\epsilon_{2}}{N_{2}} \ dV
$$
We thus want to unpack the $\mathcal{O}(\epsilon)$ terms because the $\mathcal{O}(1)$ terms directly correspond to $\int_{V}g(\phi_{1},\phi_{2})$. 

Recall that $\epsilon_{i} = \frac{R_{G,i}^{2}}{6}\nabla^{2}\phi_{i}$, we also note that $\phi_{1} = 1 - \phi_{2}$. Therefore
$$
\epsilon_{2} = \frac{R_{G,2}^{2}}{6}\nabla^{2}\phi_{2} = -\frac{R_{G,2}^{2}}{6}\nabla^{2}\phi_{1}
$$
Factoring out $\nabla^{2}\phi_{1}$, 
$$
\int_{V} \frac{1}{6}\left(\frac{R_{G,2}^{2}}{N_{2}} \ln{(1-\phi_{1})} - \frac{R_{G,1}^{2}}{N_{1}}\ln{\phi_{1}}  +\frac{R_{G,2}^{2}}{N_{2}} - \frac{R_{G,1}^{2}}{N_{1}}\right)\nabla^{2}\phi_{1}
$$

#### Helpful Aside 2: Integration by Parts and Divergence Theorem

Recall integration by parts for higher dimensions:
$$
\int_{V} u \nabla \cdot (\mathbf{V}) \ dV = \int_{S} u \mathbf{V}\cdot \mathbf{n} \ dS - \int_{V}\nabla u \cdot \mathbf{V} \ dV 
$$
By specifying suitable BCs, the surface integral disappears. We also want to recall the following Vector Calc identity:
$$
\nabla^{2}\phi = \nabla \cdot (\nabla\phi)
$$
Back to our problem. 
$$
\int_{V} \ln{(1-\phi_{1})} \nabla^{2}\phi_{1} = -\int_{V} \nabla (\ln{(1-\phi_{1})}) \cdot \nabla \phi_{1} \ dV \\
= -\int_{V} \frac{-1}{1-\phi_{1}}\nabla\phi_{1} \cdot \nabla\phi_{1} \ dV \\
= \int_{V}\frac{1}{1-\phi_{1}}(\nabla\phi_{1})^{2}
$$

$$
\int_{V}\ln{\phi_{1}} \nabla^{2}\phi_{1} = -\int_{V} \frac{1}{\phi_{1}}\nabla (\phi_{1})^{2}
$$

$$
\int_{V}\nabla^{2}\phi_{1} = \int_{V} \nabla 1 \cdot \nabla\phi_{1} = 0
$$

So our unitary terms disappear.

We thus have: 
$$
\int_{V} \frac{1}{6} \left( \frac{R_{G,2}^{2}}{N_{2}(1-\phi_{1})} + \frac{R_{G,1}^{2}}{N_{1}\phi_{1}} \right) (\nabla\phi_{1})^{2}
$$
 We thus recover
$$
\kappa_{entropic} = \frac{1}{3}\left(\frac{R_{G,1}^{2}}{N_{1}\phi_{1}} + \frac{R_{G,2}^{2}}{N_{2}(1-\phi_{1})} \right)
$$
Doing the same process for $\kappa_{enthalpic}$, 
$$
\int_{V} \chi \phi_{1}\phi_{2} \ dV
$$
Inserting our perturbation
$$
\int_{V}\chi (\phi_{1} - \epsilon_{1}) (\phi_{2} - \epsilon_{2}) \ dV \\
%
= \int_{V} \chi \phi_{1}\phi_{2} -\chi( \epsilon_{2}\phi_{1} +\epsilon_{1}\phi_{2}) \ dV \\
$$
Focusing $\mathcal{O}(\epsilon)$,
$$
-\frac{\chi}{6}\int_{V} -\phi_{1} R_{G,2}^{2}\nabla^{2}\phi_{1} + R_{G,1}^{2}(1-\phi_{1}) \nabla^{2}\phi_{1} \ dV \\
%
= \frac{\chi}{6}\int_{V}R_{G,2}^{2} \phi_{1}\nabla^{2}\phi_{1} + R_{G,1}^{2}\phi_{1}\nabla^{2}\phi_{1} - R_{G,1}^{2}\nabla^{2}\phi_{1} \ dV \\
%
= \frac{\chi}{6}\int_{V} \left(R_{G,1}^{2} + R_{G,2}^{2} \right)(\nabla \phi)^{2}
$$
By inspection, 
$$
\kappa_{enthalpic} = \frac{\chi}{3}\left(R_{G,1}^{2} + R_{G,2}^{2} \right)
$$

## Ternary

For the ternary system, the Free energy functional looks like
$$
\mathcal{G}(\phi_{1},\phi_{2},\phi_{3}) =  \\ \int_{V} g(\phi_{1},\phi_{2},\phi_{3}) + \frac{\kappa_{1}}{2}(\nabla^{2}\phi_{1}) + \frac{\kappa_{2}}{2}(\nabla^{2}\phi_{2}) + \kappa_{12}(\nabla \phi_{1})(\nabla \phi_{2}) \ dV
$$
We start with 
$$
\int_{V} \frac{\phi_{1}}{N_{1}} \ln{\phi_{1}} + \frac{\phi_{2}}{N_{2}} \ln{\phi_{2}} + \frac{\phi_{3}}{N_{3}}\ln{\phi_{3}} \ dV
$$
Inserting the perturbations
$$
\int_{V} \left( \frac{(\phi_{1} - \epsilon_{1})}{N_{1}} \ln{(\phi_{1} - \epsilon_{1})} + \frac{(\phi_{2} - \epsilon_{2})}{N_{2}} \ln{(\phi_{2} - \epsilon_{2})} \\ + \frac{(\phi_{3} - \epsilon_{3})}{N_{3}} \ln{(\phi_{3} - \epsilon_{3})}\right) \ dV
$$
Expanding and similarly retaining terms of $\mathcal{O}(\epsilon)$,
$$
\int_{V} -\frac{1}{N_{1}}\epsilon_{1}\ln{\phi_{1}} - \frac{\epsilon_{1}}{N_{1}}  - \frac{1}{N_{2}}\epsilon_{2} \ln{\phi_{2}} - \frac{\epsilon_{2}}{N_{2}} - \frac{1}{N_{3}}\epsilon_{3} \ln{\phi_{3}} - \frac{\epsilon_{3}}{N_{3}}\ dV
$$
Inserting our expressions for $\epsilon$, 
$$
\int_{V} \left( \frac{-1}{N_{1}}\frac{R_{G,1}^{2}}{6}\nabla^{2}\phi_{1}\ln{\phi_{1}} - \frac{R_{G,1}^{2}}{6N_{1}}\nabla^{2}\phi_{1} -\frac{1}{N_{2}}\frac{R_{G,2}^{2}}{6}\nabla^{2}\phi_{2}\ln{\phi_{2}} - \frac{R_{G,2}^{2}}{6N_{2}}\nabla^{2}\phi_{2} \\ -\frac{1}{N_{3}}\frac{R_{G,3}^{2}}{6}\nabla^{2}\phi_{3}\ln{\phi_{3}} - \frac{R_{G,3}^{2}}{6N_{3}}\nabla^{2}\phi_{3} \right) \ dV
$$
Doing this term by term
$$
\int_{V}\frac{-1}{N_{1}}\frac{R_{G,1}^{2}}{6}\nabla^{2}\phi_{1}\ln{\phi_{1}} - \frac{R_{G,1}^{2}}{6N_{1}}\nabla^{2}\phi_{1} \ dV \\
%
= \int_{V} \frac{R_{G,1}^{2}}{6} \frac{1}{N_{1}\phi_{1}}(\nabla \phi_{1})^{2} \ dV
$$

$$
\int_{V} -\frac{1}{N_{2}}\frac{R_{G,2}^{2}}{6}\nabla^{2}\phi_{2}\ln{\phi_{2}} - \frac{R_{G,2}^{2}}{6N_{2}}\nabla^{2}\phi_{2} \ dV \\
%
= \int_{V} \frac{R_{G,2}^{2}}{6}\frac{1}{N_{2}\phi_{2}} (\nabla\phi_{2})^{2}
$$

$$
\int_{V} -\frac{1}{N_{3}}\frac{R_{G,3}^{2}}{6}\nabla^{2}\phi_{3}\ln{\phi_{3}} - \frac{R_{G,3}^{2}}{6N_{3}}\nabla^{2}\phi_{3} \ dV \\
%
\text{Dropping the second term because it disappears}\\
= -\frac{R_{G,3}^{2}}{6N_{3}}\int_{V}\nabla^{2}(1-\phi_{1}-\phi_{2}) \ln{(1-\phi_{1} - \phi_{2})} \ dV \\
%
= \frac{R_{G,3}^{2}}{6N_{3}}\int_{V} \frac{1}{1-\phi_{1}-\phi_{2}} \left( (\nabla \phi_{1})^{2} + (\nabla\phi_{2})^{2} + 2\nabla \phi_{1}\cdot \nabla \phi_{2} \right)
$$



Adding things up: 
$$
\int_{V} \frac{1}{6}\left(\frac{R_{G,1}^{2}}{N_{1}\phi_{1}} + \frac{R_{G,3}^{2}}{N_{3}(1-\phi_{1}-\phi_{2})} \right) (\nabla \phi_{1})^{2} + \\
%
\frac{1}{6}\left( \frac{R_{G,2}^{2}}{N_{2}\phi_{2}} + \frac{R_{G,3}^{2}}{N_{3}(1-\phi_{1}-\phi_{2})} \right) (\nabla \phi_{2})^{2} + \\
%
\frac{1}{3}\frac{R_{G,3}^{2}}{N_{3}} \frac{1}{1-\phi_{1}-\phi_{2}} (\nabla \phi_{1}\cdot \nabla \phi_{2}) \ dV
$$
We thus get
$$
\kappa_{1, entropic} = \frac{1}{3}\left(\frac{R_{G,1}^{2}}{N_{1}\phi_{1}} + \frac{R_{G,3}^{2}}{N_{3}(1-\phi_{1}-\phi_{2})} \right)
$$

$$
\kappa_{2,entropic} = \frac{1}{3}\left( \frac{R_{G,2}^{2}}{N_{2}\phi_{2}} + \frac{R_{G,3}^{2}}{N_{3}(1-\phi_{1}-\phi_{2})} \right)
$$

$$
\kappa_{12,entropic} = \frac{1}{3}\frac{R_{G,3}^{2}}{N_{3}} \frac{1}{1-\phi_{1}-\phi_{2}}
$$



For the Enthalpic component,
$$
\int_{V} \chi_{12}\phi_{1}\phi_{2} + \chi_{13}\phi_{1}\phi_{3} + \chi_{23}\phi_{2}\phi_{3}
$$
Introducing our perturbation
$$
\int_{V} \chi_{12}(\phi_{1}-\epsilon_{1})(\phi_{2}-\epsilon_{2}) + \chi_{13}(\phi_{1}-\epsilon_{1})(\phi_{3}-\epsilon_{3}) + \chi_{23}(\phi_{2}-\epsilon_{2})(\phi_{3}-\epsilon_{3}) \ dV
$$
Expanding out and retain terms of $\mathcal{O}(\epsilon)$ and below,
$$
\int_{V} \chi_{12}\left( \phi_{1}\phi_{2} -\epsilon_{2}\phi_{1} - \epsilon_{1}\phi_{2} \right) + \chi_{13}(\phi_{1}\phi_{3}-\epsilon_{3}\phi_{1} - \epsilon_{1}\phi_{3}) \\
+ \chi_{23}(\phi_{2}\phi_{3} - \epsilon_{3}\phi_{2}-\epsilon_{2}\phi_{3}) 
$$
$$\mathcal{O}(\epsilon)$$,
$$
\int_{V} -\chi_{12}\frac{R_{G,2}^{2}}{6}\phi_{1}\nabla^{2}\phi_{2} - \chi_{12}\frac{R_{G,1}^{2}}{6}\phi_{2}\nabla^{2}\phi_{1} \\
%
- \chi_{13}\frac{R_{G,3}^{2}}{6}\phi_{1}\nabla^{2}\phi_{3} -\chi_{13}\frac{R_{G,1}^{2}}{6}\phi_{3}\nabla^{2}\phi_{1} \\
%
-\chi_{23}\frac{R_{G,3}^{2}}{6}\phi_{2}\nabla^{2}\phi_{3} - \chi_{23}\frac{R_{G,2}^{2}}{6}\phi_{3} \nabla^{2}\phi_{2} \ dV
$$
Expanding out $\phi_{3}$ terms
$$
\int_{V} -\chi_{12}\frac{R_{G,2}^{2}}{6}\phi_{1}\nabla^{2}\phi_{2} - \chi_{12}\frac{R_{G,1}^{2}}{6}\phi_{2}\nabla^{2}\phi_{1} \\
%
-\chi_{13}\frac{R_{G,3}^{2}}{6} \phi_{1}\left(-\nabla^{2}\phi_{1} - \nabla^{2}\phi_{2}\right) - \chi_{13}\frac{R_{G,1}^{2}}{6}(1-\phi_{1}-\phi_{2})\nabla^{2}\phi_{1} \\
%
-\chi_{23}\frac{R_{G,3}^{2}}{6}\phi_{2}(-\nabla^{2}\phi_{1} - \nabla^{2}\phi_{2}) - \chi_{23}\frac{R_{G,2}^{2}}{6}(1-\phi_{1}-\phi_{2})\nabla^{2}\phi_{2} \ dV
$$
Dropping constant coefficient terms because they disappear
$$
\int_{V} -\chi_{12}\frac{R_{G,2}^{2}}{6}\phi_{1}\nabla^{2}\phi_{2} - \chi_{12}\frac{R_{G,1}^{2}}{6}\phi_{2}\nabla^{2}\phi_{1} \\
%
+ \chi_{13}\frac{R_{G,3}^{2}}{6}\phi_{1} \nabla^{2}\phi_{1} + \chi_{13}\frac{R_{G,3}^{2}}{6}\phi_{1}\nabla^{2}\phi_{2} \\
%
\chi_{13}\frac{R_{G,1}^{2}}{6}\phi_{1}\nabla^{2}\phi_{1} + \chi_{13}\frac{R_{G,1}^{2}}{6}\phi_{2}\nabla^{2}\phi_{1} \\
%
+\chi_{23}\frac{R_{G,3}^{2}}{6}\phi_{2} \nabla^{2}\phi_{1} + \chi_{23}\frac{R_{G,3}^{2}}{6}\phi_{2}\nabla^{2}\phi_{2} \\
%
+ \chi_{23}\frac{R_{G,2}^{2}}{6}\phi_{1}\nabla^{2}\phi_{2} + \chi_{23}\frac{R_{G,2}^{2}}{6}\phi_{2}\nabla^{2}\phi_{2} \ dV
$$
Grouping terms
$$
\int_{V} \left(\chi_{13}\frac{R_{G,3}^{2}}{6}  + \chi_{13}\frac{R_{G,1}^{2}}{6}\right) \phi_{1}\nabla^{2}\phi_{1} \\
%
+ \left( \chi_{23}\frac{R_{G,3}^{2}}{6} + \chi_{23}\frac{R_{G,2}^{2}}{6} \right)\phi_{2}\nabla^{2}\phi_{2} \\
%
+ \left( -\chi_{12}\frac{R_{G,2}^{2}}{6} + \chi_{13}\frac{R_{G,3}^{2}}{6} + \chi_{23}\frac{R_{G,2}^{2}}{6} \right)\phi_{1}\nabla^{2}\phi_{2} \\
%
+ \left( -\chi_{12}\frac{R_{G,1}^{2}}{6} + \chi_{13}\frac{R_{G,1}^{2}}{6} \chi_{23}\frac{R_{G,3}^{2}}{6} \right) \phi_{2}\nabla^{2}\phi_{1} \ dV
$$
Integrating by parts and getting our $\kappa$
$$
\kappa_{1,enthalpic} = \frac{\chi_{13}}{3}( R_{G,1}^{2} + R_{G,3}^{2})
$$

$$
\kappa_{2,enthalpic} = \frac{\chi_{23}}{3}(R_{G,2}^{2} + R_{G,3}^{2})
$$

$$
\kappa_{12,enthalpic} = \frac{1}{6}\left( (R_{G,1}^{2} + R_{G,3}^{2})\chi_{13} + (R_{G,2}^{2} + R_{G,3}^{2})\chi_{23} \\ -(R_{G,1}^{2} + R_{G,2}^{2})\chi_{12} \right)
$$

## Some Comments

- Using the perturbation approach outlined here, we are able to generate the $\kappa_{enthalpic}$ terms correctly
  - See https://doi.org/10.1002/(SICI)1099-0488(20000515)38:10%3C1301::AID-POLB50%3E3.0.CO;2-M, https://doi.org/10.1002/polb.1990.09028121
- However, the entropic terms are a bit off from the same works: 
  - For the binary mixture, in one paper, the expression matches correctly: https://doi.org/10.1002/polb.1989.090271306
  - However for the polymer-solvent and polymer-polymer-solvent systems, they do not match up. 
- The method used by Ariyapadi (https://doi.org/10.1002/polb.1990.090281216) only introduces $N-1$ $\epsilon$ terms, which means that information (i.e., $R_{G,N}$ )about the $Nth$ species is not captured which leaves me suspicious. 



