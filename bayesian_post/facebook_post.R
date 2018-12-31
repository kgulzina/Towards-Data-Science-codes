library(rstan)
library(ggplot2)
library(ilog)
library(reshape)

# Read the dataset:
d <- read.csv("dataset_Facebook.csv", sep = ";")

# Extract variables of interest:
d <- d[,c(2, 5, 16, 17, 18, 19)]

# Descriptive data analysis:
str(d)
head(d)
tapply(d$Total.Interactions, d$Type, mean)
tapply(d$Total.Interactions, d$Post.Weekday, mean)


# Histogram plots for the visual descriptive analysis:
# histogram of total interactions
ggplot(d, aes(x = Total.Interactions, y=..density..)) + 
    geom_histogram(binwidth = 60, col = "black", fill = "white") + 
    theme_bw() + xlim(0, 2500) + ylab("density") + xlab("") +
    ggtitle("Total Interactions")

# histogram of total interations by type 
ggplot(d, aes(x = Total.Interactions, y=..density..)) + 
    geom_histogram(binwidth = 50, col = "black", fill = "white") + 
    xlim(0, 2500) + xlab("") +
    theme_bw() + facet_grid(. ~ Type)
# histogram of total interations by weekday
ggplot(d, aes(x = Total.Interactions, y=..density..)) + 
    geom_histogram(binwidth = 50, col = "black", fill = "white") + 
    xlim(0, 2000) + xlab("") +
    theme_bw() + facet_grid(.~ Post.Weekday)

# histogram of total interations by type and weekday (not included in the final report)
ggplot(d, aes(x = Total.Interactions, y=..density..)) + 
    geom_histogram(binwidth = 50, col = "black", fill = "white") + 
    xlim(0, 2200) +
    theme_bw() + facet_grid(Post.Weekday ~ Type)


# From the data visualization, we see that our data is highly skewed. 
# Hence we see it right to log transform the data as a remedy: 
# *Note: data contains 6 zero observations of total.interactions. 
# They have been replaced by reasonably "small" values using "ilog" R package.
d$y <- ilog(d$Total.Interactions)


# Histogram plots for the visual descriptive analysis of log transformed data:
# histogram of total interactions
ggplot(d, aes(x = y, y = ..density..)) + 
    geom_histogram(binwidth = 0.2, col = "black", fill = "white") + 
    theme_bw() + xlim(0,10) + xlab("") + ylab("density") + 
    ggtitle("log(Total Interactions)")

# histogram of total interactions by type
ggplot(d, aes(x = y, y=..density..)) + 
    geom_histogram(binwidth = 0.5 ,col = "black", fill = "white") + 
    theme_bw() + facet_grid(. ~ Type) + xlab("")
# histogram of total interactions by weekday
ggplot(d, aes(x = y, y=..density..)) + 
    geom_histogram(binwidth = 0.5 ,col = "black", fill = "white") + 
    theme_bw() + facet_grid(. ~ Post.Weekday) + xlab("")


# Now, after choosing the appropriate priors and setting up the hierarchical
# model. we can sample from the posterior distributions using MCMC:
# index for "Type" categorical variable
d$type_id <- as.numeric(d$Type)

model = "
data  {
int<lower = 0> N;  // data
int<lower = 1, upper = 4> type_id[N]; 
int<lower = 1, upper = 7> day_id[N];
real y[N];
real a;
real b;
matrix [7,7] Dw;  
matrix [7,7] W;
}

parameters {
vector[4] beta;
vector[7] delta;
real<lower = a, upper = b> rho;
real<lower = 0> sigma;
}

transformed parameters {
vector[N] mu;
matrix[7,7] Omega;
for (i in 1:N)
mu[i] = beta[type_id[i]] + delta[day_id[i]];  
Omega = Dw - rho*W;
}

model {
rho ~ uniform(a,b);
delta ~ multi_normal_prec([0,0,0,0,0,0,0], Omega); 
sigma ~ cauchy(0,1);
y ~ normal(mu, sigma);
}

"


Dw <- diag(rep(2,7), ncol = 7)
W = matrix(c(0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,
             1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,0), ncol = 7)

dat = list(y = d$y, N = nrow(d), Dw = Dw, W = W, 
           b = 1/eigen(solve(sqrt(Dw))%*%W%*%solve(sqrt(Dw)))$values[1],
           a = 1/eigen(solve(sqrt(Dw))%*%W%*%solve(sqrt(Dw)))$values[7],
           day_id <- d$Post.Weekday, type_id <- d$type_id)

m = stan_model(model_code = model)
r = sampling(mm, dat, c("beta", "delta", "rho", "sigma"), iter = 10000)


# Now, we can analyze the posterior (simulations sampled from joint 
# posterior distribution of parameters):
# posterior estimates
r 

# ci level: 0.8 (80% intervals)
# outer level: 0.95 (95% intervals)
plot(r, pars=c('beta', 'delta', 'rho', 'sigma'))

# extract samples
samples <- extract(r)


# Visual analysis of posteriors:
# prapare data to plot histograms/lines
posterior_rho <- data.frame(posterior = matrix(samples$rho))
prior_rho <- matrix(runif(20000, min = 1/eigen(solve(sqrt(Dw))%*%W%*%solve(sqrt(Dw)))$values[7],
                          max = 1/eigen(solve(sqrt(Dw))%*%W%*%solve(sqrt(Dw)))$values[1]), ncol = 1)
rhot <- cbind(posterior_rho, prior_rho)
rhol <- melt(rhot, variable_name = "distribution")

# prior vs. posterior rho line
ggplot(rhol, aes(x = value, y = ..density.., linetype = distribution)) +
    geom_density() + ggtitle("Rho") + xlab("")

# posterior rho histogram
ggplot(rhot, aes(x = posterior, y = ..density..)) + 
    geom_histogram(bins = 70, fill = "white",
                   col = "black") + theme_bw() + xlim(-2,2) + ylab("density") + xlab("") + ggtitle("Posterior distribution of Rho")

# posterior sigma histogram
sigma <- data.frame(sigma = matrix(samples$sigma))
ggplot(sigma, aes(x = sigma)) + geom_histogram(aes(y = ..density..), 
                                               bins = 70, fill = "white", col = "black") + theme_bw() + 
    ggtitle("Posterior distribution of Sigma")


# Visual analysis of posterior Beta (precisely):
beta <- data.frame(samples$beta)
colnames(beta) <- c("beta1", "beta2", "beta3", "beta4")
betal <- melt(beta, variable_name = "parameter")
# all together beta's
ggplot(betal, aes(x = value, y = ..density.., colour = parameter)) +
    geom_density() + ggtitle("Posterior beta") + xlab("") + xlim(-2,10)


# Visual analysis of posterior Delta (precisely):
delta <- data.frame(samples$delta)
colnames(delta) <- c("Monday", "Tuesday", "Wednesday", "Thursday", 
                     "Friday", "Saturday", "Sunday")
deltal <- melt(delta, variable_name = "parameter")
ggplot(deltal, aes(x = value, y = ..density.., colour = parameter)) +
    geom_density() + ggtitle("Posterior delta") + xlab("") + xlim(-5,5)


# Analyze posterior means separately:
calc_post_mean <- function(beta, delta, type){
    # Calculates posterior means for specific "Type" and "Weekdays" variables
    # 
    # Args:
    #   beta:  Samples of posterior beta  
    #   delta: Samples of posterior delta    
    #   type:  Type variable, given as integer(int) number. 1:Link, 2:Photo, 
    #          3:Status, 4:Video
    #    
    # Returns: data frame of posterior mean simulations
    
    # create an empty data frame
    df <- as.data.frame(matrix(NA, ncol=7, nrow=nrow(beta)))
    
    for (i in 1:7){
        df[,i] <- beta[,type] + delta[,i]
    }
    
    return(df)
}

mu1 <- calc_post_mean(beta, delta, 1)
mu2 <- calc_post_mean(beta, delta, 2)
mu3 <- calc_post_mean(beta, delta, 3)
mu4 <- calc_post_mean(beta, delta, 4)

# combine all together
mu <- cbind(mu1, mu2, mu3, mu4)
colnames(mu) <- c("Mon_Link","Tue_Link", "Wed_Link", "Thu_Link", "Fri_Link", "Sat_Link", "Sun_Link", "Mon_Photo","Tue_Photo", "Wed_Photo", "Thu_Photo", "Fri_Photo", "Sat_Photo", "Sun_Photo", "Mon_Status","Tue_Status", "Wed_Status", "Thu_Status", "Fri_Status", "Sat_Status", "Sun_Status", "Mon_Video","Tue_Video", "Wed_Video", "Thu_Video", "Fri_Video", "Sat_Video", "Sun_Video")

# get the descriptive analysis of posterior marginals
for (i in c(1,8,15,22)) {
    print(apply(mu[,i:(i+6)], MARGIN = 2, quantile, c(0.25, 0.5, 0.975)))
}


# reshape into long
mul <- melt(mu, variable_name = "Day_type")

# Plot of posterior marginal means
ggplot(mul, aes(x=value, y=..density.., colour=Day_type)) +
    geom_density() + xlab("") +
    ggtitle("Posterior marginal means")


# Finally, we calculate the probabilities as follows:
prob <- function(mu) {
    df = as.data.frame(matrix(NA, nrow = 1, ncol = 28))
    colnames(df) <- colnames(mu)
    for (i in 1:28){
        df[1,i] <- sum(mu[,i]>mu[,-i])/(27*nrow(mu))
    }
    return(df)
}

probs <- prob(mu)
probs
