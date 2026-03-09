# Homework 1
6 march 2026

## Map coloring

#### Problem description
- Model the problem as a CSP and solve the above instance.
- Change the model so that Germany and Denmark are always the same color. Except for Germany and Denmark, no two neighboring countries are the same color.
- Include Switzerland in the map: Switzerland shares borders with France and Germany.

### Solution
- Create variable for each country and assign it a domain of colors.
- Add constraints that ensure no two neighboring countries have the same color.
- Add constraint that Germany and Denmark must have the same color.
- Add constraint that Switzerland must have a different color from France and Germany.
- Use `CpSolver` to find a valid coloring.

## Blocked N-Queens
- Model the problem as a CSP and solve a 4x4 instance.
- Consider a list of blocked positions. Model the problem and test on instances.

### Solution
- Create function to generate a list of blocked positions randomly
- Overwrite the solver's printer function to print the board nicely
- Create a list of variables for each column which will store the row of the queen in that column
- Add constraints that ensure no two queens attack each other on the same row and diagonals
- Add constraints that ensure no queen is placed on a blocked position
- Use `CpSolver` to find a valid solution