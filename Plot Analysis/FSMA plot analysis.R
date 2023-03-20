library(purrr)
library(data.table)
library(ggplot2)

training_1 <- data.table::fread(file = file.path(getwd(), "1_anthracyclineTaxaneChemotherapy_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_2 <- data.table::fread(file = file.path(getwd(), "10_skinPsoriatic_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_3 <- data.table::fread(file = file.path(getwd(), "2_brainTumour_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_4 <- data.table::fread(file = file.path(getwd(), "3_BurkittLymphoma_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_5 <- data.table::fread(file = file.path(getwd(), "4_gingivalPeriodontits_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_6 <- data.table::fread(file = file.path(getwd(), "5_heartFailurFactors_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_7 <- data.table::fread(file = file.path(getwd(), "6_hepatitisC_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_8 <- data.table::fread(file = file.path(getwd(), "7_humanGlioma_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_9 <- data.table::fread(file = file.path(getwd(), "8_ovarianTumour_training.csv"), header = TRUE, 
                                sep=',', na.strings="?", stringsAsFactors = FALSE)

training_10 <- data.table::fread(file = file.path(getwd(), "9_septicShock_training.csv"), header = TRUE, 
                                 sep=',', na.strings="?", stringsAsFactors = FALSE)

Lista_DataSet <- list(training_1, training_2, training_3, training_4, training_5, training_6, training_7, training_8, training_9, training_10)

wykres <-function(i)
{
  pca <- prcomp( Lista_DataSet[[i]][,-ncol(Lista_DataSet[[i]]),with = FALSE], center=TRUE, scale=TRUE)
  fig1 <- ggplot(data.table(pca$x),aes(x=PC1,y=PC2,
                                       colour = Lista_DataSet[[i]][1:nrow(Lista_DataSet[[i]]), target]))+
    geom_point(aes(shape =Lista_DataSet[[i]][1:nrow(Lista_DataSet[[i]]), target]), size = 1)+
    labs(x = "PC1", y = "PC2", colour = "target", shape = "target", title = "Data visualization")
  bar_plot <- suppressWarnings(ggplot(data=data.table(id = 1:6, std = pca$sdev), aes(x=id, y=std))+
                                 geom_bar(stat="identity", position=position_dodge())+
                                 labs(title = "Comparison of standard deviation"))
  print(fig1)
  print(bar_plot)
}

wykres(1)

wykres(2)

wykres(3)

wykres(4)

wykres(5)

wykres(6)

wykres(7)

wykres(8)

wykres(9)

wykres(10)
