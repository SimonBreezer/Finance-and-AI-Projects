
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

Install Packages
```{r}


##  Install packages
list.of.packages <- c("shiny","dplyr","DT","readr","readxl","tidyr", "RcppRoll","ggfortify","doParallel","SHAPforxgboost","fpc","e1071",
                      "randomForest","xgboost", "lime","viridis", "DiagrammeR", "devtools", "gganimate")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

devtools::install_github('cttobin/ggthemr')
```

Load Packages

```{r}

library(data.table)
library(ggplot2)
library(GGally)
library(caret)
library(mlbench)
library(corrplot)
library(tidyverse)
library(ggthemes)
library(tidytext)
library(tidyr)
library(yarrr)  #Pirate plot
library(formattable) #For the color_tile function
library(kableExtra) #Create nicely formatted output tables
library(reticulate)
library(xgboost)
library(plotly)
library(lime)
library(readr)

```

Load file
```{r}

clean_df <- read_csv("Cleaned_Titles_Csv.csv", na = c("", "NA", "#N/A"))

```

Data Cleaning and Conditioning
```{r}
# Remove non full cycle Titles
x <- clean_df$TITLENAME
clean_df$remove <- ifelse(x %in%  c("DRIVECLUB", "BEYOND: Two Souls PS4"), 1, 0)
clean_df <- subset(clean_df, remove==0)

# Convert Dates to Posix Date which is easy to work with
clean_df$Alpha_WSR <- as.POSIXct(strptime(clean_df$Alpha_WSR, format= "%Y-%m-%d"))
clean_df$Beta_WSR <- as.POSIXct(strptime(clean_df$Beta_WSR, format= "%Y-%m-%d"))
clean_df$FormatQASubmission_WSR <- as.POSIXct(strptime(clean_df$FormatQASubmission_WSR, format= "%Y-%m-%d"))

# Create function to calculate months between dates
elapsed_months <- function(end_date, start_date) {
  ed <- as.POSIXlt(end_date)
  sd <- as.POSIXlt(start_date)
  12 * (ed$year - sd$year) + (ed$mon - sd$mon)
}

# Calculate months between dates using the above function
clean_df$MONTHS_TO_ALPHA <- elapsed_months(as.Date(clean_df$Alpha_WSR),as.Date(clean_df$ORIG_DATE))
clean_df$MONTHS_TO_BETA <- elapsed_months(as.Date(clean_df$Beta_WSR),as.Date(clean_df$ORIG_DATE))
clean_df$MONTHS_TO_QASUBMISSION <- elapsed_months(as.Date(clean_df$FormatQASubmission_WSR),as.Date(clean_df$ORIG_DATE))

# Split the dataframe to Functional and Localisation
func_df <- subset(clean_df, DEPARTMENT_C=="FUNCTIONALITY")
loc_df <- subset(clean_df, DEPARTMENT_C=="LOCALISATION")
```

Rollup Functionality Data Frames

```{r}
## Calculate total hours for functionality
tot_func_hrs <- func_df %>% 
  group_by(TITLENAME,PLATFORM,GENRE,STUDIO,FIRST_RELEASE_DATE,FIRST_RELEASE_YEAR,FIRST_RELEASE_MONTH,VR,
           MULTI_PLATFORM,Genre_eedar,Gameplay_area_eedar,Online_eedar,Multiplayer_eedar,Combat_speed_eedar,Sequel,Game_Origin_US, Size) %>%
  summarise(MAX_MTH_TO_REL = max(MONTHS_TO_RELEASE),
            MAX_DAYS_TO_REL = max(DAYS_TO_RELEASE),
            FUNC_TOT_HRS=sum(HOURS),
            PRE_REL_HRS = sum(HOURS[is_POST_RELEASE==0]),
            POST_REL_HRS = sum(HOURS[is_POST_RELEASE==1])) %>%
  ungroup()

## Calculate monthly hours for functionality
mon_func_hrs <- func_df %>% 
  group_by(TITLENAME,PLATFORM,GENRE,STUDIO,FIRST_RELEASE_DATE,MONTHS_TO_RELEASE,MONTHS_TO_ALPHA,MONTHS_TO_BETA,MONTHS_TO_QASUBMISSION,VR, PORTED,
           MULTI_PLATFORM,Genre_eedar,Gameplay_area_eedar,Online_eedar,Multiplayer_eedar,Combat_speed_eedar,Sequel,Game_Origin_US, Size) %>%
  summarise(MAX_MTH_TO_REL = max(MONTHS_TO_RELEASE),
            MAX_DAYS_TO_REL = max(DAYS_TO_RELEASE),
            FUNC_TOT_HRS=sum(HOURS),
            PRE_REL_HRS = sum(HOURS[is_POST_RELEASE==0]),
            POST_REL_HRS = sum(HOURS[is_POST_RELEASE==1]),
            COUNT = n()) %>%
  ungroup()

# Remove non-full cycle titles
x <- mon_func_hrs$TITLENAME
mon_func_hrs$remove <- ifelse(x %in%  c("inFAMOUS First Light", "Nioh 2","EVE: Valkyrie","Helldivers",
                                        "Air Force Special Ops: Nightfall","Battlezone","BigFest","Counterspy","TRACK LAB"), 1, 0)
mon_func_hrs <- subset(mon_func_hrs, remove==0)

#Write csv to local folder
write.csv(tot_func_hrs, file = "tot_func_hrs.csv")
write.csv(mon_func_hrs, file = "mon_func_hrs.csv")

```


Data Preparation before running models

```{r}
# Convert factor columns to factors
cols <- c("PLATFORM","GENRE","STUDIO","Genre_eedar","Gameplay_area_eedar","Combat_speed_eedar","Game_Origin_US","Size")
tot_func_hrs[cols] <- lapply(tot_func_hrs[cols], factor)

#Remove columns that are not required
x1 <- tot_func_hrs %>% select(-TITLENAME,-GENRE,-FIRST_RELEASE_DATE,-PRE_REL_HRS,-POST_REL_HRS,-STUDIO,-FIRST_RELEASE_YEAR,-FIRST_RELEASE_MONTH,-MAX_DAYS_TO_REL)

# Create dummy variables (Binary 0/1)
dummy <- dummyVars(" ~ .", data=x1)
tot_fin_func_df <- data.frame(predict(dummy, newdata = x1))
tot_fin_func_df$FUNC_TOT_HRS <- log(tot_fin_func_df$FUNC_TOT_HRS)

#recombine Titlename columns
TITLE <- as.data.frame(tot_func_hrs$TITLENAME)
tot_fin_func_df <- data.frame(Title = TITLE,tot_fin_func_df)
tot_fin_func_df <- tot_fin_func_df %>% rename(Titlename = tot_func_hrs.TITLENAME)

```

Total Functional Model Data Prep
```{r}
# Set seed to replicate results
set.seed(123)
# trainindex <- createDataPartition(tot_fin_func_df$FUNC_TOT_HRS, p=0.80, list= FALSE)
# tr_tot_fn <- tot_fin_func_df[trainindex, ]
# te_tot_fn <- tot_fin_func_df[-trainindex, ]


tr_tot_fn <- subset(tot_fin_func_df, !(Titlename %in%  c("Concrete Genie","Drawn 2 Death","Ghost of Tsushima", 
                                                         "Gravity Daze 2 PS4", 
                                                         "Kill Strain","LocoRoco 2 Remastered","MATTERFALL","No Heroes Allowed! VR",
                                                         "Ratchet and Clank PS4", "RIGS","The Last Of Us 2","TRACK LAB")))

te_tot_fn <- subset(tot_fin_func_df, Titlename %in%  c("Concrete Genie","Drawn 2 Death","Ghost of Tsushima", 
                                                       "Gravity Daze 2 PS4", 
                                                       "Kill Strain","LocoRoco 2 Remastered","MATTERFALL","No Heroes Allowed! VR",
                                                       "Ratchet and Clank PS4", "RIGS","The Last Of Us 2","TRACK LAB"))

```


Total Functional Model
```{r}
ctrl <- trainControl(
  method="repeatedcv", # cross validation
  number=5, # 5-fold
  repeats = 10, # 10 times
  allowParallel = TRUE,
  verboseIter = FALSE # Verbose output
)

set.seed(123)
xgb_tot_fn <- train(FUNC_TOT_HRS ~., data = tr_tot_fn %>% select(-Titlename), 
                    method = "xgbTree",
                    trControl = ctrl,
                    na.action = na.pass,
                    verbose=FALSE)

summary(xgb_tot_fn)
xgb_tot_fn$bestTune
plot(xgb_tot_fn)
xgb_tot_fn
varImp(xgb_tot_fn)
plot(varImp(xgb_tot_fn), main='Variable Importance for Boosting Model')

# Make predictions
forecast_tot_fn_xgb <- as.data.frame(predict(xgb_tot_fn, te_tot_fn %>% select(-Titlename)))

pred_tot_fn_xgb <- cbind(te_tot_fn$FUNC_TOT_HRS,forecast_tot_fn_xgb)
pred_tot_fn_xgb<-cbind(te_tot_fn %>% select(Titlename),pred_tot_fn_xgb)

names(pred_tot_fn_xgb)[2] <- "Log_Actual"
names(pred_tot_fn_xgb)[3] <- "Log_Forecast"

pred_tot_fn_xgb$Actual <- exp(pred_tot_fn_xgb$Log_Actual)
pred_tot_fn_xgb$Forecast <- exp(pred_tot_fn_xgb$Log_Forecast)

## Remove log columns
pred_tot_fn_xgb <- pred_tot_fn_xgb %>% select(Titlename,Actual, Forecast)


## Plot Graph

library(reshape2)
c<-melt(pred_tot_fn_xgb,id.vars="Titlename")

ggplot(c,aes(x=variable,y=value,fill = variable, colour = variable))+
  facet_wrap(~Titlename,scales = "free") + 
  geom_bar(stat="identity",position="dodge") +
  labs(title = "Total Functional Hours", 
       x="Actual vs Forecast", 
       y="Functional Total Hours")

```

Monthly Functional Model

```{r}

## Plot monthly functional hours for all titles

z <- mon_func_hrs %>% ungroup() %>% arrange(transform(Size))

z <- arrange(transform(z,TITLENAME=factor(TITLENAME,levels=unique(z$TITLENAME))),TITLENAME)

# z <- subset(z, TITLENAME %in%  c("Guns Up","Blood & Truth","Kill Strain", 
#                                                        "The Last Of Us 2"))

ggplot(data = z,
       aes(x = MAX_MTH_TO_REL, y = FUNC_TOT_HRS)) +
  geom_line(aes(colour = Size), size = 1.5) +
  facet_wrap(~TITLENAME, scales = "free") +
  scale_x_reverse() + 
  labs(title = "Title Monthly Functional Hours", 
       x="Months to Release | 0 is the release month", 
       y="Functional Total Hours")

## Create dummy variable data
cols <- c("PLATFORM","GENRE","STUDIO","Genre_eedar","Gameplay_area_eedar","Combat_speed_eedar","Game_Origin_US","Size")
mon_func_hrs[cols] <- lapply(mon_func_hrs[cols], factor)

x2 <- mon_func_hrs %>% select(-TITLENAME,-GENRE,-FIRST_RELEASE_DATE,-PRE_REL_HRS,-POST_REL_HRS,-STUDIO,-MAX_DAYS_TO_REL,-COUNT)
write.csv(x2, file = "x2.csv")

dummy <- dummyVars(" ~ .", data=x2)

mon_fin_func_df <- data.frame(predict(dummy, newdata = x2))

mon_fin_func_df$FUNC_TOT_HRS <- log(mon_fin_func_df$FUNC_TOT_HRS)

TITLE <- as.data.frame(mon_func_hrs$TITLENAME)
mon_fin_func_df <- data.frame(Title = TITLE,mon_fin_func_df)
mon_fin_func_df <- mon_fin_func_df %>% rename(Titlename = mon_func_hrs.TITLENAME)

tr_mon_fn <- subset(mon_fin_func_df, !(Titlename %in%  c("Concrete Genie","Drawn 2 Death","Ghost of Tsushima", 
                                                         "God of War 3 Remastered", "Here They Lie",
                                                         "Kill Strain","LocoRoco 2 Remastered","MATTERFALL",
                                                         "Ratchet and Clank PS4", "RIGS","The Last Of Us 2","Tumble VR")))

te_mon_fn <- subset(mon_fin_func_df, Titlename %in%  c("Concrete Genie","Drawn 2 Death","Ghost of Tsushima", 
                                                       "God of War 3 Remastered", "Here They Lie",
                                                       "Kill Strain","LocoRoco 2 Remastered","MATTERFALL",
                                                       "Ratchet and Clank PS4", "RIGS","The Last Of Us 2","Tumble VR"))
ctrl <- trainControl(
  method="repeatedcv", # cross validation
  number=5, # 5-fold
  repeats = 10, # 10 times
  allowParallel = TRUE,
  verboseIter = FALSE # Verbose output
)

set.seed(123)
xgb_mon_fn <- train(FUNC_TOT_HRS ~., data = tr_mon_fn %>% select(-Titlename, -MAX_MTH_TO_REL), 
                      method = "xgbTree",
                      trControl = ctrl,
                      na.action = na.pass,
                      verbose=FALSE)

summary(xgb_mon_fn)
xgb_mon_fn$bestTune
xgb_mon_fn
plot(xgb_mon_fn)
varImp(xgb_mon_fn)
plot(varImp(xgb_mon_fn),main='Variable Importance for Boosting Model')

forecast_mon_fn_xgb <- as.data.frame(predict(xgb_mon_fn, te_mon_fn %>% select(-Titlename, -MAX_MTH_TO_REL)))

pred_mon_fn_xgb <- cbind(te_mon_fn$FUNC_TOT_HRS,forecast_mon_fn_xgb)

z <- subset(mon_func_hrs, TITLENAME %in%  c("Concrete Genie","Drawn 2 Death","Ghost of Tsushima", 
                                            "God of War 3 Remastered", "Here They Lie",
                                            "Kill Strain","LocoRoco 2 Remastered","MATTERFALL",
                                            "Ratchet and Clank PS4", "RIGS","The Last Of Us 2","Tumble VR"))

pred_mon_fn_xgb<-cbind(z %>% select(TITLENAME,Size,PLATFORM,Genre_eedar,MONTHS_TO_RELEASE,Size),pred_mon_fn_xgb)

names(pred_mon_fn_xgb)[6] <- "Log_Actual"
names(pred_mon_fn_xgb)[7] <- "Log_Forecast"

pred_mon_fn_xgb$Actual <- exp(pred_mon_fn_xgb$Log_Actual)
pred_mon_fn_xgb$Forecast <- exp(pred_mon_fn_xgb$Log_Forecast)

pred_mon_fn_xgb <- subset(pred_mon_fn_xgb, TITLENAME %in%  c("Ratchet and Clank PS4","RIGS","MATTERFALL", 
                                 "The Last Of Us 2"))

library(ggthemr)
#ggthemr("solarized")
ggthemr("flat")

### Plot Test Graph
ggplotly(
  ggplot(data = pred_mon_fn_xgb, aes(x = MONTHS_TO_RELEASE)) +
    geom_line(aes(y = Actual,colour = Size),size = 0.8) +
    geom_line(aes(y = Forecast),size = 0.7, color = "black",linetype="dash") + 
    facet_wrap(~TITLENAME, scales = "free") +
    scale_x_reverse() +
    labs(title = "Predicted Monthly Functional hours for a title", 
         x="Months to Release | 0 is the Release Month", 
         y="Functional Efforts in Hours"))

# to remove all ggthemr effects later:
ggthemr_reset()

```


Localisation Model
Data Prep

```{r}
##Inputation for wordcount, languages and playthrough_time for localization
loc_df <- loc_df %>% group_by(Size) %>%
  mutate(OST_wordcount=ifelse(is.na(OST_wordcount),round(mean(OST_wordcount,na.rm=TRUE)),OST_wordcount),
         VO_wordcount=ifelse(is.na(VO_wordcount),round(mean(VO_wordcount,na.rm=TRUE)),VO_wordcount),
         Total_wordcount=ifelse(is.na(Total_wordcount),round(mean(Total_wordcount,na.rm=TRUE)),Total_wordcount),
         Languages=ifelse(is.na(Languages),round(mean(Languages,na.rm=TRUE)),Languages),
         Playthrough_time=ifelse(is.na(Playthrough_time),round(mean(Playthrough_time,na.rm=TRUE),1),Playthrough_time))

## Remove MLB from Localisation dataset
x <- loc_df$TITLENAME
loc_df$remove <- ifelse(x %in%  c("MLB '15: The Show","MLB '16: The Show","MLB The Show 17","MLB The Show 18", 
                                  "MLB The Show 19", "MLB The Show 20"), 1, 0)
loc_df <- subset(loc_df, remove==0)

## Calculate total hours for localisation
tot_loc_hrs <- loc_df %>% 
  group_by(TITLENAME,PLATFORM,Size,GENRE,STUDIO,FIRST_RELEASE_DATE,VR, PORTED,
           MULTI_PLATFORM,Genre_eedar,Gameplay_area_eedar,Online_eedar,Multiplayer_eedar,Combat_speed_eedar,Sequel,Game_Origin_US,OST_wordcount,
           VO_wordcount,Total_wordcount,Languages,Playthrough_time) %>%
  summarise(MAX_MTH_TO_REL = max(MONTHS_TO_RELEASE),
            MAX_DAYS_TO_REL = max(DAYS_TO_RELEASE),
            LOC_TOT_HRS=sum(HOURS),
            PRE_REL_HRS = sum(HOURS[is_POST_RELEASE==0]),
            POST_REL_HRS = sum(HOURS[is_POST_RELEASE==1]),
            n_records = n()) %>%
  ungroup()


## Calculate monthly hours for localisation
mon_loc_hrs <- loc_df %>% 
  group_by(TITLENAME,PLATFORM,Size,GENRE,STUDIO,FIRST_RELEASE_DATE,MONTHS_TO_RELEASE,MONTHS_TO_ALPHA,MONTHS_TO_BETA,MONTHS_TO_QASUBMISSION,VR, PORTED,
           MULTI_PLATFORM,Genre_eedar,Gameplay_area_eedar,Online_eedar,Multiplayer_eedar,Combat_speed_eedar,Sequel,Game_Origin_US,OST_wordcount,
           VO_wordcount,Total_wordcount,Languages,Playthrough_time) %>%
  summarise(MAX_MTH_TO_REL = max(MONTHS_TO_RELEASE),
            MAX_DAYS_TO_REL = max(DAYS_TO_RELEASE),
            LOC_TOT_HRS=sum(HOURS),
            PRE_REL_HRS = sum(HOURS[is_POST_RELEASE==0]),
            POST_REL_HRS = sum(HOURS[is_POST_RELEASE==1]),
            n_records = n()) %>%
  ungroup()

densityplot(mon_loc_hrs$LOC_TOT_HRS)




#bins <- cut(func_hrs$FUNC_TOT_HRS, 5, include.lowest = TRUE)

write.csv(tot_loc_hrs, file = "tot_loc_hrs.csv")
write.csv(mon_loc_hrs, file = "mon_loc_hrs.csv")

```

Visualize Model
```{r}
z <- mon_loc_hrs %>% ungroup() %>% arrange(transform(Size))
z <- arrange(transform(z,TITLENAME=factor(TITLENAME,levels=unique(z$TITLENAME))),TITLENAME)

ggplot(data = z,
       aes(x = MAX_MTH_TO_REL, y = LOC_TOT_HRS)) +
  geom_line(aes(colour = Size), size = 1.5) +
  facet_wrap(~TITLENAME, scales = "free") +
  scale_x_reverse() + 
  labs(title = "Title Monthly Localisation Hours", 
       x="Months to Release | 0 is the release month", 
       y="Localisation Total Hours")

```



Total Localisation Model

```{r}

tr_tot_lc <- subset(tot_fin_loc_df, !(Titlename %in%  c("Concrete Genie","Gran Turismo Sport", "Drawn 2 Death",
                                                        "MATTERFALL","Here They Lie","The Last Of Us 2")))

te_tot_lc <- subset(tot_fin_loc_df, Titlename %in%  c("Concrete Genie","Gran Turismo Sport", "Drawn 2 Death",
                                                      "MATTERFALL","Here They Lie","The Last Of Us 2"))

ctrl <- trainControl(
  method="repeatedcv", # cross validation
  number=5, # 10-fold
  repeats = 10, # 5 times
  allowParallel = TRUE,
  verboseIter = FALSE # Verbose output
)

set.seed(123)
xgb_tot_lc <- train(LOC_TOT_HRS ~., data = tr_tot_lc %>% select(-Titlename, -MAX_MTH_TO_REL), 
                    method = "xgbTree",
                    trControl = ctrl,
                    na.action = na.pass,
                    verbose=FALSE)

summary(xgb_tot_lc)
xgb_tot_lc$bestTune
plot(xgb_tot_lc)
xgb_tot_lc
varImp(xgb_tot_lc)
plot(varImp(xgb_tot_lc), main='Variable Importance for Boosting Model')

forecast_tot_lc_xgb <- as.data.frame(predict(xgb_tot_lc, te_tot_lc %>% select(-Titlename)))

pred_tot_lc_xgb <- cbind(te_tot_lc$LOC_TOT_HRS,forecast_tot_lc_xgb)
pred_tot_lc_xgb<-cbind(te_tot_lc %>% select(Titlename),pred_tot_lc_xgb)

names(pred_tot_lc_xgb)[2] <- "Log_Actual"
names(pred_tot_lc_xgb)[3] <- "Log_Forecast"

pred_tot_lc_xgb$Actual <- exp(pred_tot_lc_xgb$Log_Actual)
pred_tot_lc_xgb$Forecast <- exp(pred_tot_lc_xgb$Log_Forecast)

## Remove log columns
pred_tot_lc_xgb <- pred_tot_lc_xgb %>% select(Titlename,Actual, Forecast)


## Plot Graph

library(reshape2)
c<-melt(pred_tot_lc_xgb,id.vars="Titlename")

ggplot(c,aes(x=variable,y=value,fill = variable, colour = variable))+
  facet_wrap(~Titlename,scales = "free") + 
  geom_bar(stat="identity",position="dodge") +
  labs(title = "Total Localisation Hours", 
       x="Actual vs Forecast", 
       y="Localisation Total Hours")
```


Monthly Localisation Model

```{r}
#### Start monthly localisation modelling
cols <- c("PLATFORM","GENRE","STUDIO","Genre_eedar","Gameplay_area_eedar","Combat_speed_eedar","Game_Origin_US","Size")
mon_loc_hrs[cols] <- lapply(mon_loc_hrs[cols], factor)

x4 <- mon_loc_hrs %>% select(-TITLENAME,-GENRE,-FIRST_RELEASE_DATE,-PRE_REL_HRS,-POST_REL_HRS,-STUDIO,-MAX_DAYS_TO_REL,-n_records)

dummy <- dummyVars(" ~ .", data=x4)

mon_fin_loc_df <- data.frame(predict(dummy, newdata = x4))

mon_fin_loc_df$LOC_TOT_HRS <- log(mon_fin_loc_df$LOC_TOT_HRS)

TITLE <- as.data.frame(mon_loc_hrs$TITLENAME)
mon_fin_loc_df <- data.frame(Title = TITLE,mon_fin_loc_df)
mon_fin_loc_df <- mon_fin_loc_df %>% rename(Titlename = mon_loc_hrs.TITLENAME)

#densityplot(fin_func_df$FUNC_TOT_HRS)
#densityplot(fin_func_df$FUNC_TOT_HRS)

M<-cor(mon_fin_loc_df %>% select(-Titlename))
round(M,2)

set.seed(123)
#trainindex <- createDataPartition(mon_fin_func_df$FUNC_TOT_HRS, p=0.80, list= FALSE)

tr_mon_lc <- subset(mon_fin_loc_df, !(Titlename %in%  c("Animal Force","Ghost of Tsushima", "The Last Of Us 2","The Last Guardian",
                                                    "Blood & Truth", "Frantics")))

te_mon_lc <- subset(mon_fin_loc_df, Titlename %in%  c("Bloodborne", "Nioh 2","Animal Force",
                                                  "Uncharted 4", "Alienation", "The Last Of Us 2"))



ctrl <- trainControl(
  method="repeatedcv", # cross validation
  number=5, # 10-fold
  repeats = 10, # 5 times
  allowParallel = TRUE,
  verboseIter = FALSE # Verbose output
)

set.seed(123)
xgb_mon_lc <- train(LOC_TOT_HRS ~., data = tr_mon_lc %>% select(-Titlename, -MAX_MTH_TO_REL), 
                      method = "xgbTree",
                      trControl = ctrl,
                      na.action = na.pass,
                      verbose=FALSE)

summary(xgb_mon_lc)
xgb_mon_lc$bestTune
plot(xgb_mon_lc)
xgb_mon_lc
varImp(xgb_mon_lc)
plot(varImp(xgb_mon_lc), main='Variable Importance for Boosting Model')

forecast_mon_lc_xgb <- as.data.frame(predict(xgb_mon_lc, te_mon_lc %>% select(-Titlename, -MAX_MTH_TO_REL)))

pred_mon_lc_xgb <- cbind(te_mon_lc$LOC_TOT_HRS,forecast_mon_lc_xgb)

z <- subset(mon_loc_hrs, TITLENAME %in%  c("Bloodborne", "Nioh 2","Animal Force",
                                            "Uncharted 4", "Alienation", "The Last Of Us 2"))

pred_mon_lc_xgb<-cbind(z %>% select(TITLENAME,Size,PLATFORM,Genre_eedar,MONTHS_TO_RELEASE,Size),pred_mon_lc_xgb)

names(pred_mon_lc_xgb)[6] <- "Log_Actual"
names(pred_mon_lc_xgb)[7] <- "Log_Forecast"

pred_mon_lc_xgb$Actual <- exp(pred_mon_lc_xgb$Log_Actual)
pred_mon_lc_xgb$Forecast <- exp(pred_mon_lc_xgb$Log_Forecast)

## Remove log columns
pred_mon_lc_xgb <- pred_mon_lc_xgb %>% select(TITLENAME,MONTHS_TO_RELEASE,Size, Actual, Forecast)

#write.csv(pred_lc_xgb, file = "pred_mon_lc_xgboost_log.csv")


### Plot Test Graph
library(ggthemr)
#ggthemr("solarized")
ggthemr("flat")

ggplotly(
  ggplot(data = pred_mon_lc_xgb, aes(x = MONTHS_TO_RELEASE)) +
    geom_line(aes(y = Actual,colour = Size),size = 0.8) +
    geom_line(aes(y = Forecast),size = 0.7, color = "black",linetype="dash") + 
    facet_wrap(~TITLENAME, scales = "free") +
    scale_x_reverse() +
    labs(title = "Monthly Localisation hours for a title", 
         x="Months to Release | 0 is the Release Month", 
         y="Localization Efforts in Hours")
)

# to remove all ggthemr effects later:
ggthemr_reset()

```

New Title Predictions

```{r}
################################################################
###### Predictions for a new title
############################################################


nt_df <- read_csv("new_title.csv")
nt_om_df <- read_csv("new_title_om.csv")


pred_nt_fn <- as.data.frame(predict(xgb_mon_fn, nt_df))
pred_nt_fn <- cbind(nt_df %>% select(MONTHS_TO_RELEASE),pred_nt_fn)
names(pred_nt_fn)[2] <- "Log_Forecast_Loc"

pred_nt_fn$Forecast <- round(exp(pred_nt_fn$Log_Forecast_Loc),2)

write.csv(pred_nt_fn, file = "pred_fn_new_title.csv")

pred_nt_om_fn <- as.data.frame(predict(xgb_mon_fn, nt_om_df))
pred_nt_om_fn <- cbind(nt_om_df %>% select(MONTHS_TO_RELEASE),pred_nt_om_fn)
names(pred_nt_om_fn)[2] <- "Log_Forecast_Loc"


##Combine Forecasts
#pred_nt_fn$Forecast2 <- round(exp(pred_nt_om_fn$Log_Forecast_Loc),2)
pred_nt_fn <- pred_nt_fn %>% select(-Log_Forecast_Loc)



nt_df_lc <- read_csv("new_title_loc.csv")

pred_nt_lc <- as.data.frame(predict(xgb_mon_lc, nt_df_lc))
pred_nt_lc <- cbind(nt_df_lc %>% select(MONTHS_TO_RELEASE),pred_nt_lc)
names(pred_nt_lc)[2] <- "Log_Forecast_Loc"

pred_nt_lc$Forecast <- round(exp(pred_nt_lc$Log_Forecast_Loc),2)

write.csv(pred_nt_lc, file = "pred_lc_new_title.csv")

##Combine Forecasts
#pred_nt_fn$Forecast2 <- round(exp(pred_nt_om_fn$Log_Forecast_Loc),2)
pred_lc_fn <- pred_lc_fn %>% select(-Log_Forecast_Loc)

# set ggthemr theme
library(ggthemr)
#ggthemr("solarized")
ggthemr("flat")

ggplotly(
  ggplot(data = pred_nt_fn, aes(x = MONTHS_TO_RELEASE)) +
    geom_line(aes(y = Forecast), size = 1, color = "#00AFBB", linetype="dash") +
    #geom_line(aes(y = Forecast2), size = 1, color = "#FC4E07", linetype="dash") +
    scale_x_reverse()+
    #scale_y_continuous(name = "Frequency of Incidents", breaks = 1000) +
    labs(title = "Forecast: Functional hours for a new title", 
         x="Months to Release | 0 is the Release Month", 
         y="Functional Efforts in Hours")
)

```

check SHAP Values

```{r}
library("SHAPforxgboost")
y_var <-  "LOC_TOT_HRS"
dataX_lc <- tr_mon_fn %>% select(-Titlename, -MAX_MTH_TO_REL, -FUNC_TOT_HRS)
# hyperparameter tuning results


mod_lc <- xgboost::xgboost(data = as.matrix(dataX_lc), label = as.matrix(tr_mon_fn$FUNC_TOT_HRS), 
                        max.depth = 3, eta = 0.3, subsample = 1, colsample_bytree = 0.6,
                        min_child_weight = 1, nrounds = 150,
                        nthread = parallel::detectCores() - 2, 
                        objective = "reg:squarederror",verbose = 1)

importance_matrix_lc <- xgb.importance(model = mod_lc)
print(importance_matrix_lc)
xgb.plot.importance(importance_matrix = importance_matrix_lc)

forecast_lc_xgb2 <- as.data.frame(predict(mod_lc,as.matrix(te_lc %>% select(-LOC_TOT_HRS,-Titlename, -MAX_MTH_TO_REL))))

pred_lc_xgb2 <- cbind(te_lc$LOC_TOT_HRS,forecast_lc_xgb2)

te_data <- subset(mon_loc_hrs, TITLENAME %in%  c("Animal Force","Ghost of Tsushima", "The Last Of Us 2","The Last Guardian",
                                                 "Blood & Truth", "Frantics"))

pred_lc_xgb2<-cbind(te_data %>% select(TITLENAME,Size,PLATFORM,Genre_eedar,MONTHS_TO_RELEASE,Size),pred_lc_xgb2)

names(pred_lc_xgb2)[6] <- "Actual"
names(pred_lc_xgb2)[7] <- "Forecast"

pred_lc_xgb2$Actual <- exp(pred_lc_xgb2$Actual)
pred_lc_xgb2$Forecast <- exp(pred_lc_xgb2$Forecast)

# To return the SHAP values and ranked features by mean|SHAP|
shap_values <- shap.values(xgb_model = mod_lc, X_train = dataX_lc)
# The ranked features by mean |SHAP|
shap_values$mean_shap_score

# To prepare the long-format data:
shap_long <- shap.prep(xgb_model = mod_lc, X_train = dataX_lc)
# is the same as: using given shap_contrib
shap_long <- shap.prep(shap_contrib = shap_values$shap_score, X_train = dataX_lc)
# (Notice that there will be a data.table warning from `melt.data.table` due to `dayint` coerced from integer to double)

# **SHAP summary plot**
shap.plot.summary(shap_long)

# sometimes for a preview, you want to plot less data to make it faster using `dilute`
shap.plot.summary(shap_long, x_bound  = 3.5, dilute = 8)


shap.plot.summary.wrap1(mod_lc, X = dataX_lc)


# option 2: supply a self-made SHAP values dataset (e.g. sometimes as output from cross-validation)
shap.plot.summary.wrap2(shap_values$shap_score, dataX_lc)


xgb.plot.shap(data=as.matrix(dataX_lc),model = mod_lc,top_n = 15, n_col = 3)
```


