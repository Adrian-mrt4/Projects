
## Project Overview

The goal of this program is to assign a set of tasks with specific durations to a fixed number of identical parallel machines. The objective is to minimize the **makespan** (the time when the last machine finishes its work).

Instead of using a greedy approach, this project utilizes the A\* algorithm to explore the state space and find an optimal or near-optimal schedule based on a custom heuristic function.

## Key Features

  * A* Algorithm Implementation: Uses a priority queue (min-heap) to explore the most promising states first based on $f(n) = g(n) + h(n)$.
  * Custom Heuristic h(n): Implements a look-ahead heuristic that estimates the remaining cost by calculating the average load required per machine minus the current available "free space" (gaps) in the schedule.
  * State Management: Utilizes a `closed_list` (implemented via `std::map`) to detect visited states and prevent cycles or redundant processing.
  * **Performance Metrics:** Measures and outputs the total search time and the final makespan.

## Technical Details

  * Language: C++ (Standard Template Library)
  * Data Structures:
      * `std::priority_queue`: For the Open List (managing the search frontier).
      * `std::map`: For the Closed List (managing visited states).
      * `structs`: Custom structures for Machines, Tasks, and System States.
