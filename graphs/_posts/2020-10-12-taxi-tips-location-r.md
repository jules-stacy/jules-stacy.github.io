---
title: "Taxi Tip Amounts vs Geocoordinates"
categories:
  - graphs
tags:
  - graphs
  - taxi
  - r
  - ggplot2
---

![tip amount vs pickup location new york]({{ https://jules-stacy.github.io/ }}/assets/images/green_taxi.png "a graph of pickup locations by geocoordinates, with tip amount mapped to color scale, and overlaid on top of a map of new york city")

A standalone exercise in R, this graph is composed of a scatterplot of pickup locations in geocoordinates. The color of these points is then mapped to the amount received in tips, and the scatterplot is then overlaid on top of a map of New York City.
Ggplot2 was used to construct the scatterplot. Ggmaps was used to plot this on top of the map, which was obtained from Open Street Maps through a package called osmdata.

## Code Walkthrough

Prior to modeling, an exploratory data analysis was performed and numerous graphs were generated.  


