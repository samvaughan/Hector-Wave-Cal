data{
    int<lower=1> N; // Number of data points
    int<lower=1> N_fibres; // Number of fibres
    int<lower=1> N_predictors; // Number of fibres
    vector[N] wavelengths;
    array[N] int fibre_numbers;
    matrix[N, N_predictors] predictors;
    real wavelengths_std;
    real wavelengths_mean;

}

parameters{

    matrix[N_fibres, N_predictors] a;
    vector[N_fibres] constants;
    real<lower=0> sigma;
}


model{

    vector[N] mu;
    for (i in 1:N){
        mu[i] = constants[fibre_numbers[i]] + dot_product(a[fibre_numbers[i]], predictors[i]);
    }   
    wavelengths ~ normal(mu, sigma);

}

generated quantities {
    array[N] real wavelengths_ppc;
    for (i in 1:N){
        wavelengths_ppc[i] = normal_rng(constants[fibre_numbers[i]] + dot_product(a[fibre_numbers[i]], predictors[i]), sigma);
        wavelengths_ppc[i] = wavelengths_ppc[i] * wavelengths_std + wavelengths_mean;
    }
}