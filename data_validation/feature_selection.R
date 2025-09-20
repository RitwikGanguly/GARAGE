## the R version  => 4.5.1
## the R packages => matrixStats, plyr, resample

## strategy => Here, we followed the FS method as, FS from real data applied in real data, FS from gen data applied to real data 
##             and FS from combined data (gen + real) applied to real data.



library(matrixStats)


Fano_ind<-function(PData){
	    library(resample)
    bdata=as.matrix(PData)
        Fano_factor = colVars(bdata)/colMeans(bdata)
        ID = order(Fano_factor)
	    SelectedGenes_ID = ID[1:100]
	    return(SelectedGenes_ID)
	        }

CV2<-function(PData){
	#library(dplyr)
	library(plyr)
	library(data.table)
	#library('edgeR')
	    
	get_variable_gene<-function(m) {
		  
		  df<-data.frame(mean=colMeans(m),cv=apply(m,2,sd)/colMeans(m),var=apply(m,2,var))
	  df$dispersion<-with(df,var/mean)
	    df$mean_bin<-with(df,cut(mean,breaks=c(-Inf,quantile(mean,seq(0.1,1,0.05)),Inf)))
	    var_by_bin<-ddply(df,"mean_bin",function(x) {
				          data.frame(bin_median=median(x$dispersion),
						                    bin_mad=mad(x$dispersion))
					    })
	      df$bin_disp_median<-var_by_bin$bin_median[match(df$mean_bin,var_by_bin$mean_bin)]
	      df$bin_disp_mad<-var_by_bin$bin_mad[match(df$mean_bin,var_by_bin$mean_bin)]
	        df$dispersion_norm<-with(df,abs(dispersion-bin_disp_median)/bin_disp_mad)
	        df
	}


	datan=PData
	ngenes_keep = 100 #top 1000 genes
	cat("Select variable Genes...\n")
	df<-get_variable_gene(datan)
	gc()
	cat("Sort Top GenViewes...\n")
	disp_cut_off<-sort(df$dispersion_norm,decreasing=T)[ngenes_keep]
	cat("Cutoff Genes...\n")
	df$used<-df$dispersion_norm >= disp_cut_off
	top_features = head(order(-df$dispersion_norm),ngenes_keep)
	    return(top_features)
	
    }


    ## FOR POLLEN, CBMC, MURARO
gen_data = read.csv(
    "/path/to/load/the/gen/data/pollen_data_mixdata_iter5_top_90.csv", 
    header = TRUE,
    stringsAsFactors = FALSE,
    row.names = 1
    )

# Use the Fano index function to select 100 features from gen data
selected_features_gen <- CV2(gen_data)

## FOR YAN DATA, POLLEN DATA
# Read the original processed data
org_data <- read.csv(
        "/path/to/load/the/real/data/pollen_process.txt",
        check.names = FALSE,
        stringsAsFactors = FALSE,
        header = FALSE,
        )


## FOR CBMC, MURARO
# org_data <- read.csv(
#         "/home/bernadettem/bernadettenotebook/Ritwik/NLP/GAT_GAN/real_data/pollen_process.txt",
#         check.names = FALSE,
#         stringsAsFactors = FALSE,
#         header = TRUE,
#         row.names = 1
#         )

## "YAN" & "CBMC" needs ==> transpose, "POLLEN" & "MURARO" does not need transpose
# org_data = t(org_data)

org_data = as.data.frame((org_data), stringsAsFactors = FALSE)

rownames(org_data) <- NULL

colnames(org_data) = colnames(gen_data)
rownames(org_data) = seq_len(nrow(org_data))

combined_data <- rbind(gen_data, org_data)


# Use the Fano index function to select 100 features from real data
selected_features_org <- CV2(org_data)

# Apply Fano index to combined data to select top features
selected_features_combined <- CV2(combined_data)

# Filter the original data using the selected feature indices
data_filtered1 <- org_data[, selected_features_gen]
data_filtered2 <- org_data[, selected_features_org]
data_filtered_combined <- org_data[, selected_features_combined]

# Write the filtered original data to a CSV file for subsequent ARI computation
## this files will later use in the data validation script (data_validation.R)


write.csv(data_filtered1, '/path/to/save/the/gen/data/feature_selection/datafilt1.csv', row.names = FALSE)
write.csv(data_filtered2, '/path/to/save/the/real/data/feature_selection/data_validation/datafilt2.csv', row.names = FALSE)
write.csv(data_filtered_combined, '/path/to/save/the/combined/data/feature_selection/datafilt_combined.csv', row.names = FALSE)