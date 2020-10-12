---
title: "Taxi Tip Amounts vs Geocoordinates"
categories:
  - graphs
tags:
  - graphs
  - open street maps
  - r
  - ggplot2
  - ggmap
---

A graph of taxi tip amounts based on pickup geocoordinates, programmed in R using ggplot2, ggmap, and open street maps.

![tip amount vs pickup location new york]({{ https://jules-stacy.github.io/ }}/assets/images/green_taxi.png "a graph of pickup locations by geocoordinates, with tip amount mapped to color scale, and overlaid on top of a map of new york city")

We were tasked with creating a visual that demonstrates where taxi drivers can expect to find the highest tips in New York City.

A standalone exercise in R, this graph is composed of a scatterplot of pickup locations in geocoordinates. The color of these points is mapped to the amount received in tips, and the scatterplot is then overlaid on top of a map of New York City.

The scatterplot was constructed in ggplot2 and ggmaps was used to plot this on top of the map, which was obtained from Open Street Maps through a package named osmdata.

## Code Walkthrough

To recreate the graph, we will begin by importing packages used in the plotting process.

{% highlight r %}
#install.packages("tidyverse")
#install.packages("ggplot2")
#install.packages("ggmap")
#install.packages("osmdata")

library("ggplot2")
library("tidyverse")
library("ggmap")
library("osmdata")
{% endhighlight %}

The dataset is too large to host on github so I am unable to provide it. Let's see exactly how large it is.

{% highlight r %}
dim(df)
{% endhighlight %}
```
[1] 1494926      21
```

That is a lot of cab rides! There are 1,494,926 cab rides logged in the dataset, each with its own latitude and longitude for pickup location.

There are just too many points to make a readable graph using my available hardware. A minimum tip amount was arbitrarily chosen in order to cut down on the number of observations to be graphed. There is a lot of statistical groundwork that was skipped for this portion, and ultimately this constrains any conclusions to be drawn from the graphic to comparisons between tip amounts greater than the cutoff.

{% highlight r %}
#set minimum tip cutoff
mintip = 10
df1 <- (subset(df, Tip_amount > mintip))
dim(df1)
{% endhighlight %}
```
[1] 10509    21
```

It will be a lot easier to visually represent 10.5 thousand points rather than 1.5 million points.

### Digital Pseudo-Cartography (or, Making Data-Rich Backgrounds) <br>

Now that I have the observations to be plotted, I can get the map that will serve as the background for the plot. To make it easier to zone in on the area I wish to include, I set the minimum and maximum latitude and longitude.

{% highlight r %}
#grab NYC map
#set minimum and maximum lat and long
minlong <- -74.1
maxlong <- -73.7
minlat <- 40.55
maxlat <- 40.95
{% endhighlight %}

Since OSM reads latitude and longitude in matrix format, the desired variables need to be converted into a matrix and the rows and columns need to be properly named. 

{% highlight r %}
#create a matrix out of min/max lat/long
mapbox=matrix(c(minlong, minlat, maxlong, maxlat), nrow=2, ncol=2)

#give the matrix rows and columns the proper names
rownames(mapbox) <- c("x","y")
colnames(mapbox) <- c("min","max")

#fetch the boxed area using osmdata
nyc_map <- get_map(mapbox, maptype = "terrain", zoom = 11)
{% endhighlight %}

And finally osmdata is called using the get_map() function. This returns a map of new york city which is stored in the variable nyc_map.


### Building a Graph<br>

The final step is to construct the graph. This is done by painstakingly layering elements and adjusting aesthetics until you realize that the deadline is 20 minutes away and you finally convince yourself that the graph looks great and you won't push your luck (but let's be real, it could *always* look better). In other words, all the prepwork is done and it is time to actually do the heavy lifting. Details below.

{% highlight r %}
#set the caption for the graph
capt1 = "Map data from Open Street Map \n Tip data from green_tripdata_2015-09.csv"
{% endhighlight %}

The caption which details the sources of the data is stored in a variable. This will later be displayed on the graph (lower right corner).


## Building the Graph<br>
First I will show you the entire chunk for the graph, then I will explain the purpose of each piece.
{% highlight r %}
#set the map of nyc as the background for the graph
p <- ggmap(nyc_map)+

  #create the scatterplot
  geom_point(data=df1 %>%
    arrange(Tip_amount), 
    aes(x = Pickup_longitude,
      y = Pickup_latitude,
      color=Tip_amount,
      alpha=Tip_amount,
      size=Tip_amount)
      ) +
  
  #set a theme (so that it can be modified later
  theme_bw() +
  
  #set the bounds of the graph
  xlim(minlong, maxlong) +
  ylim(minlat, maxlat)+

  #fine tune the alpha and point size mapped to the data
  scale_alpha_continuous(guide="none", range = c(.4, 1))+
  scale_size_continuous(guide="none")+

  #adjust the color gradient
  scale_color_gradient2(
    low="#4b028e",
    mid='#04c3bf',
    high='#ff00cb',
    midpoint=115,
    breaks=c(10,100,200,300),
    limits=c(10, 300))+
  
  #adjust graph texts
  labs(title="New York City",
    subtitle="Green Taxi Tip Amount by Geo-Coordinates",
    caption=capt1,
    x="Longitude",
    y="Latitude")

{% endhighlight %}


Breakdown begin! (everything is fine....)


{% highlight r %}
#set the map of nyc as the background for the graph
p <- ggmap(nyc_map)+
{% endhighlight %}

To begin, the new york city map is called and set as the graph background using ggmap(). I find that ggplot2 and ggmap do not play well together unless ggmap is called first.

{% highlight r %}
  #create the scatterplot
  geom_point(data=df1 %>%
    arrange(Tip_amount), 
    aes(x = Pickup_longitude,
      y = Pickup_latitude,
      color=Tip_amount,
      alpha=Tip_amount,
      size=Tip_amount)
      ) +
{% endhighlight %}

Next the scatterplot is created using ggplot2. The data is called from the dataframe df1 and arranged by tip amount. This is done because the points are plotted in the order they are fed into the scatterplot. And because there are so many datapoints being plotted, the largest observations would get buried underneath smaller observations if they were plotted first.

After arranging the plotting order, the x and y axes are set for longitude and latitude. 

Finally, the color, alpha, and size aesthetics for observations are mapped to the tip amount for each observation. This was done to differentiate tip amounts from each other. It was not enough to differentiate based on color, so size and alpha also had to be mapped.

{% highlight r %}  
  #set a theme (so that it can be modified later
  theme_bw() +
{% endhighlight %}

A theme had to be set in order to adjust values within the theme. Why does ggplot2 require this, and why can't it use default values? The world may never know. Moving on:

{% highlight r %}  
  #set the bounds of the graph
  xlim(minlong, maxlong) +
  ylim(minlat, maxlat)+
{% endhighlight %}

The bounds for the graph are set to match the map of nyc. This is done because the cab data extends past the bounds of the requested map, and if the bounds were not specified there would be white space surrounding the map. 

"But why not grab a bigger map for your background," you may ask. The answer lies with data scarcity at boundaries: I wanted to focus the graph on pertinent data with regards to the highest tip amounts and biggest groupings of tips. In short, showing all of the data would mean having to zoom the graph out and would detract from the primary purpose of the graph.

{% highlight r %}
  #fine tune the alpha and point size mapped to the data
  scale_alpha_continuous(guide="none", range = c(.4, 1))+
  scale_size_continuous(guide="none")+
{% endhighlight %}

The alpha and size gradients are mapped to Tip_amount in an above code block and as a result they both grow in size and become more opaque (or less transparent) as the tip amount grows.

In this code block, the start and end points for the alpha scale are set. scale_size_continuous() is called purely to turn the legend off (which is done by passing guide="none").

{% highlight r %}
  #adjust the color gradient
  scale_color_gradient2(
    low="#4b028e",
    mid='#04c3bf',
    high='#ff00cb',
    midpoint=115,
    breaks=c(10,100,200,300),
    limits=c(10, 300))+
{% endhighlight %}

In this code block, the color gradient for the scatterplot is fine-tuned.

In preliminary drafts of this graph, I divided the dataset into two dataframes: one with tip amounts $10-$149, and the other with tip amounts $150 to $300. Ultimately I decided to go with a three-color gradient (or in ggplot terms, scale_color_gradient2). The low, mid, and high colors were deliberately chosen to show as much contrast as possible from each other, and to be perfectly honest I am not satisfied with the results. I feel that some of the points get washed out around the $200 range and hidden in the graph. I tried to prevent this as much as possible by adjusting color hues and the midpoint, which I found appeared best around $115. The breaks and limits portion of this code displays the given values on the legend for the graph. Note that $10 and $300 are passed into both the breaks and limits arguments.

{% highlight r %}
  #adjust graph texts
  labs(title="New York City",
    subtitle="Green Taxi Tip Amount by Geo-Coordinates",
    caption=capt1,
    x="Longitude",
    y="Latitude")
{% endhighlight %}

Lastly my favorite part, the text elements of the graph. The variable capt1 that was set earlier is called and displayed as the caption for the graph. 

My goal with this text was to convey maximally relevant information while minimizing word usage. Rather than go with "Scatterplot" as the title, I recognized that the key visual focus of the graph was the map of New York City. The subtitle conveys the subject matter of the scatterplot and axes in as few as 7 words (this could be reduced to 5 by removing "Green Taxi").