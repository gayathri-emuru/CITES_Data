#Analyze CITES COP participant data

##############
####Set Up####
##############

#clear memory
rm( list=ls() )

#Load some packages
library(tm)
library(tidyverse)
library(readxl)
library(stringr)
library(rvest)
library(quanteda)
library(XML)
library(xml2)
library(doBy)
library(reshape2)
library(readxl)
library(ggplot2)
library(tidyr)
library(dplyr)
library(sf)
library(rnaturalearth)
library(rnaturalearthdata)
library(combinat)
library(RColorBrewer)
library(wordcloud)

#set seed
set.seed(50)

#set working directory
setwd("")

###################################
#########Read in Final Data########
###################################

#read in data 
main.data <- read.csv("cites.cops.csv",encoding="UTF-8")

#######################################
#########Descriptives Main Data########
#######################################

# Get missing percentages
missing_pct <- colSums(is.na(main.data)) / nrow(main.data) * 100

# View results
missing_pct

# Collapse data in R
collapse1 <- summaryBy(Female+Male+Party+Observer~CoP+CoPyear, FUN=c(mean,sd), data=main.data)

# sort by COP year
collapse1 <- collapse1[order(collapse1$CoPyear),]

# Pivot Female.mean and Male.mean into long format directly from collapse1
collapse1_long <- collapse1 %>%
  pivot_longer(cols = c(Female.mean, Male.mean),
               names_to = "Variable",
               values_to = "Mean")

#male/female plot
pdf(file = "MF.pdf", width = 8, height = 5)  # adjust size as needed
  
ggplot(collapse1_long, aes(x = CoPyear, y = Mean, color = Variable)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_color_manual(values = c("Female.mean" = "#e41a1c", "Male.mean" = "#377eb8"),
                     labels = c("Female", "Male"),
                     name = "Variable") +
  labs(x = "Year", y = "Proportion of Attendees") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_text(size = 11),
    legend.text  = element_text(size = 10)
  )

dev.off()

#now the same for party/observer
collapse1_long <- collapse1 %>%
  pivot_longer(cols = c(Party.mean, Observer.mean),
               names_to = "Variable",
               values_to = "Mean") %>%
  # Set factor levels to match order of colors and labels below
  mutate(Variable = factor(Variable, levels = c("Party.mean", "Observer.mean")))

pdf(file = "PO.pdf", width = 8, height = 5)  # adjust size as needed
  
ggplot(collapse1_long, aes(x = CoPyear, y = Mean, color = Variable)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_color_manual(
    values = c("Party.mean" = "#4daf4a", "Observer.mean" = "#377eb8"),
    labels = c("Party", "Observer"),
    name = "Variable"
  ) +
  labs(x = "Year", y = "Proportion of Attendees") +
  theme_minimal(base_size = 14) +
  theme(
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  )

dev.off()

##################################################################################################################
####################################################Plotting Maps#################################################
##################################################################################################################

world <- ne_countries(scale = "medium", returnclass = "sf")

points_sf <- main.data %>%
  filter(!is.na(Latitude) & !is.na(Longitude)) %>%
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326)

points_joined <- st_join(points_sf, world["name"])

counts <- points_joined %>%
  st_drop_geometry() %>%
  count(name)

world_counts <- left_join(world, counts, by = "name") %>%
  mutate(n = ifelse(is.na(n), 0, n))


pdf(file = "ShadedMap.pdf", width = 10, height = 6.5)  
  
ggplot(world_counts) +
  geom_sf(aes(fill = n), color = "gray70", size = 0.1) +
  scale_fill_gradient(
    low = "lightgray",
    high = "darkblue",
    na.value = "lightgray",
    name = "Attendees",
    trans = "sqrt",
    breaks = c(0, 50, 200, 500, 1000, 1500),  # specify legend ticks you want
    labels = c("0", "50", "200", "500", "1000", "1500")  # labels for ticks
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    panel.grid.major = element_line(color = "transparent"),
    plot.title = element_blank(),
    plot.subtitle = element_blank()
  )

dev.off()

# Load world map polygons
world <- ne_countries(scale = "medium", returnclass = "sf")

# Filter out rows with missing coords
clean_data <- main.data %>%
  filter(!is.na(Latitude), !is.na(Longitude))


pdf(file = "worldpoints.pdf", width = 10, height = 6.5)  

ggplot() +
  geom_sf(data = world, fill = "white", color = "black", size = 0.3) +
  geom_jitter(data = clean_data,
              aes(x = Longitude, y = Latitude),
              width = 0.5,
              height = 0.5,
              size = 0.8,               # smaller points
              color = "#1E90FF",       # brighter blue (DodgerBlue)
              alpha = 0.5) +           # slightly more visible
  coord_sf(expand = FALSE) +
  theme_minimal() +
  theme(panel.grid.major = element_line(color = "transparent")) +
  labs(x = "Longitude", y = "Latitude")

dev.off()


###################################WordCloud####################################

#subset to only observer
main.data<-subset(main.data,Observer==1)



# Extract and clean the Delegation text vector
delegation_text <- main.data$Delegation %>%
  na.omit() %>%                  # Remove NAs
  tolower() %>%                   # Convert to lowercase for uniformity
  gsub("[()]", "", .)  

# Split into words 
words <- str_split(delegation_text, "\\s+") %>% unlist()

# Remove very common stopwords 
stopwords <- c(stopwords::stopwords("en"),"de","-")  
words <- words[!words %in% stopwords]

# Calculate word frequencies
word_freq <- table(words) %>% sort(decreasing = TRUE)

# Make the wordcloud
set.seed(123)  # for reproducibility

png("delegation_wordcloud.png", width = 1000, height = 700, res = 150)
wordcloud(names(word_freq), freq = word_freq,
          scale = c(3.5, 0.375),       # smaller max and min font sizes
          max.words = 300,
          colors = RColorBrewer::brewer.pal(8, "Dark2"),
          random.order = FALSE,
          rot.per = 0.3)
dev.off()
