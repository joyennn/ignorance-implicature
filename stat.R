library(lme4)
library(lmerTest)
library(dplyr)

data <- read.csv("result.csv")



###############################
### Result for experiment 1 ###
###############################

# convert to a categorical variable
data$context <- as.factor(data$context)
data$modifier <- as.factor(data$modifier)
data$entailment <- as.factor(data$entailment)
data$ex1_gpt <- as.factor(data$ex1_gpt)
data$ex1_gemini <- as.factor(data$ex1_gemini)

# reference group 
data$context <- relevel(as.factor(data$context), ref = "precise") 
data$modifier <- relevel(as.factor(data$modifier), ref = "bare")
data$entailment <- relevel(as.factor(data$entailment), ref = "bare") 

# glmer models
model1_gpt <- glmer(
  ex1_gpt ~ context * modifier + (1 | image) + (1 | text), 
  data = data,
  family = binomial(link = "logit")
)
model1_gemini <- glmer(
  ex1_gemini ~ context * modifier + (1 | image) + (1 | text), 
  data = data,
  family = binomial(link = "logit")
)

summary(model1_gpt)
summary(model1_gemini)





###############################
### Result for experiment 2 ###
###############################

# lmer models
model2_gpt <- lmer(
  ex2_gpt ~ context * modifier + (1 | image) + (1 | text),
  data = data)
model_gemini <- lmer(
  ex2_gemini ~ context * modifier + (1 | image) + (1 | text),
  data = data)

summary(model_gpt)
summary(model_gemini)





############################################
### Additional analysis for 'entailment' ###
############################################

# data filtering
filtered_data <- data %>%
  filter(modifier %in% c("superlative", "comparative"), entailment %in% c("upward", "downward"))

# reference group
filtered_data$modifier <- relevel(as.factor(filtered_data$modifier), ref = "superlative")  # quantifier 기준을 'bare'로
filtered_data$entailment <- relevel(as.factor(filtered_data$entailment), ref = "upward")  # type 기준을 'bare'로


# glmer / lmer models
entail1_gpt <- glmer(
  ex1_gpt ~ entailment * modifier + (1 | image) + (1 | text), 
  data = filtered_data, 
  family = binomial(link = "logit")
  )
entail1_gemini <- glmer(
  ex1_gemini ~ entailment * modifier + (1 | image) + (1 | text), 
  data = filtered_data, 
  family = binomial(link = "logit")
  )

summary(entail1_gpt)
summary(entail1_gemini)


entail2_gpt <- lmer(
  ex2_gpt ~ entailment * modifier + (1 | image) + (1 | text),
  data = filtered_data)

entail2_gemini <- lmer(
  ex2_gemini ~ entailment * modifier + (1 | image) + (1 | text),
  data = filtered_data)

summary(entail2_gpt)
summary(entail2_gemini)


