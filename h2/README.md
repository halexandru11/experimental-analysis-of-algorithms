# Homework 2
13 march 2026

## Multi-Knapsack Problem
The goal is to assign a set of items to multiple knapsacks such that the total weight in each knapsack $k$ does not exceed its capacity $c_k$, while maximizing the total value of the assigned items.

**Problem Formulation:**
- **Maximize:** $\sum_{k=1}^n r_k x_k$
- **Subject to:**
    - $\sum_{k=1}^n w_k x_k \le c_k$ for each knapsack $k \in \{1, \dots, n\}$.
    - $x_k \geq 0$ for each item $k \in \{1, \dots, n\}$.
    - $x_k \in \mathbb{N}$

Where $w_k$ represents the weight of item $k$, $r_k$ is the value of item $k$, $c_k$ is the weight capacity of knapsack $k$, and $x_k$ is a variable that is the number of items $k$ assigned to knapsack $k$.

## Production Plan Problem
A company produces ammoniac gas ($NH_3$) and ammonium chloride ($NH_4Cl$). The company has at its disposal $50$ units of nitrogen ($N$), $180$ units of hydrogen ($H$), and $40$ units of chlorine ($Cl$). The company makes a profit of $40$ Euros for each sale of an ammoniac gas unit and $50$ Euros for each sale of an ammonium chloride unit. Find a production plan to maximize the profits given the available stocks.