![MixSol](/docs/1.svg)


Pipetting planner for efficient combinatorial mixing of solutions. Often we want to interpolate a few stock solutions into many target mixtures. If some of these mixtures require only a tiny amount of a stock solution, the minimum volume for our pipette may limit our ability to make this solution without a serial dilution. Mixsol searches for mixing sequences that use only other target solutions as stepping stones to reach these difficult mixtures, minimizing waste.

Mixsol also has the ability to calculate the masses of solid reagents needed to make a target solution. Finally, measured amounts of solid reagents can be input to calculate the actual solution we have made.

Happy mixing!

# Examples

## Solution Mixing
Solutions are defined with the `Solution` class. Solutes and solvents are both defined by their formula, which follows the `(name1)(amount1)_(name2)(amount2)_..._(name)(amount)` format. The names do not have to correspond to elements, so you can use placeholders for units that will be mixed. An `alias` can be provided for the solution to simplify later analysis.

```
stock_solutions = [
    Solution(
        solutes='FA_Pb_I3',
        solvent='DMF9_DMSO1',
        molarity=1,
        alias='FAPI'
    ),
    Solution(
        solutes='MA_Pb_I3',
        solvent='DMF9_DMSO1',
        molarity=1,
        alias='MAPI'
    ),
]
```

This process goes for both stock and target solutions. 

```
densetargets = []
for a in np.linspace(0, 0.8, 5):
    densetargets.append(Solution(
        solutes=f"FA{a:0.3f}_MA{1-a:.3f}_Pb_I3",
        solvent="DMF9_DMSO1",
        molarity=1,
        alias=f'FA_{a:.3f}'
    ))
```

Stock and target solutions go into a `Mixer` object

```
sm = Mixer(
    stock_solutions = stock_solutions,
    targets = {
        t:60      #Solution:volume dictionary
        for t in densetargets
    })
```
which is then solved with constraints
```
sm.solve(
    min_volume=20, #minimum volume for a single liquid transfer
    max_inputs = 3 #maximum number of solutions that can be mixed to form one target
    )
```

The results can be displayed in two ways:
- plain text output of liquid transfers, in order. use of the `alias` term really simplifies this output
```
sm.print()
```
```
===== Stock Prep =====
120.00 of FAPI
180.00 of MAPI
====== Mixing =====
Distribute FAPI:
	54.00 to FA_0.600
	36.00 to FA_0.400
	30.00 to FA_0.800
Distribute MAPI:
	60.00 to FA_0.000
	36.00 to FA_0.600
	54.00 to FA_0.400
	30.00 to FA_0.200
Distribute FA_0.600:
	30.00 to FA_0.800
Distribute FA_0.400:
	30.00 to FA_0.200
```

- a graph of solution transfers. This is harder to use in practice, but can give an overview of the mixing path.
```
fig, ax = plt.subplots(figsize=(6,6))
sm.plot(ax=ax)
```
![Example Mixer.plot()](/docs/example_graph.png)

Note that the units of volume here are arbitrary. Using SI units for small volumes might cause numerical issues when solving a mixture strategy (eg you should use 10 microliters instead of 1e-5 liters). 

## Solution Preparation
Mixsol aids in determining the mass of solid reagents needed to form target solutions. We can also check the actual solution formed from recorded reagent masses. Here, the units *do* matter, and you should stick to SI units (mass in grams, volume in liters).

We define solid reagents with the `Powder` class. This requires at least a chemical formula delimited by underscores, similar to the `Solution` definition earlier. If this formula is a proper chemical formula of elements, the molar mass is calculated automatically. If not, you can pass the molar mass directly. The `calculate_molar_mass` function can be used for convenience. `alias` does the same thing it did for `Solution`.

```
from mixsol import Powder, calculate_molar_mass, Weigher

powders = [
    Powder('Cs_I'),
    Powder('Pb_I2'),
    Powder('Pb_Br2'),
    Powder('Pb_Cl2'),
    Powder(
        formula='MA_I',
        molar_mass=calculate_molar_mass('C_H6_N_I'),
        alias='MAI',
    ),
    Powder(
        formula='FA_I',
        molar_mass = calculate_molar_mass('C_H5_N2_I'),
        alias='FAI',
        )
]
```

The list of available `Powder`s is fed into a `Weigher` object

```
weigher = Weigher(
    powders=powders
)
```
which can then be used to determine powder amounts for a given volume of a target `Solution`

```
target=Solution(
    solutes='Cs0.05_FA0.8_MA0.15_Pb_I2.4_Br0.45_Cl0.15',
    solvent='DMF9_DMSO1',
    molarity=1
)

answer = weigher.get_weights(
    target,
    volume=1e-3, #in L
)
print(answer) #masses of each powder, in grams
```
```
{'Cs_I': 0.012990496098, 'Pb_I2': 0.322706258, 'Pb_Br2': 0.082576575, 'Pb_Cl2': 0.020857935, 'MAI': 0.02384543385, 'FAI': 0.1375746568}
```

Finally, we can also generate a `Solution` object by inputting a `{powder:mass}` dictionary into `Weigher`. The molarity of the output will by default be determined by the largest component amount. Passing a component or a numeric value to `norm` can control the molarity. Note that this does not affect the solution itself, just the relative values of the formula units and the overall molarity.

```
result = weigher.weights_to_solution(
    weights=answer,
    volume=1e-3, #in L
    solvent='DMF9_DMSO1',
    norm='Pb', #normalize the formula+molarity such that Pb=1
)
print(result) #result is a Solution object
```
```
1.0M Cs0.05_I2.4_Pb_MA0.15_Br0.45_Cl0.15_FA0.8 in DMF9_DMSO1
```


Read the full documentation [here](https://mixsol.readthedocs.io/en/latest/).
