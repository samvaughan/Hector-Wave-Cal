data{
    int<lower=1> N; // Number of data points
    int<lower=1> N_predictors;
    int<lower=1> N_fibres;
    vector[N] wavelengths;
    matrix[N, N_predictors] predictors;
    array[N] int fibre_number;
    real wavelengths_std;
    real wavelengths_mean;

}

parameters{

    vector[N_predictors] a;
    //vector<lower=0>[N_fibres] sigma;
    real<lower=0> sigma;
    //vector[N_fibres] constant;
    real constant;
}


model{

    vector[N] mu;
    vector[N] tau;

    a ~ std_normal();
    sigma ~ std_normal();
    constant ~ std_normal();

    for (i in 1:N){
        mu[i] = constant + predictors[i] * a;
        //tau[i] = sigma[fibre_number[i]];
    }

    wavelengths ~ normal(mu, sigma);
}

generated quantities {
    array[N] real wavelengths_ppc;
    for (i in 1:N){
        wavelengths_ppc[i] = normal_rng(constant + predictors[i] * a, sigma);
        wavelengths_ppc[i] = wavelengths_ppc[i] * wavelengths_std + wavelengths_mean;
    }
}