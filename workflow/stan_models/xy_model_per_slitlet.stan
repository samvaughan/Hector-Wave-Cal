data{
    int<lower=1> N; // Number of data points
    int<lower=1> N_predictors;
    int<lower=1> N_slitlets;
    vector[N] wavelengths;
    matrix[N, N_predictors] predictors;
    array[N] int slitlet_number;
    real wavelengths_std;
    real wavelengths_mean;

}

parameters{

    matrix[N_slitlets, N_predictors] a;
    //vector<lower=0>[N_fibres] sigma;
    //vector<lower=0>[N_slitlets] sigma;
    real<lower = 0> sigma;
    //vector[N_fibres] constant;
    vector[N_slitlets] constant;
}


model{

    vector[N] mu;
    vector[N] tau;

    to_vector(a) ~ std_normal();
    sigma ~ std_normal();
    constant ~ std_normal();

    // for (i in 1:N){
    //     mu[i] = constant[slitlet_number[i]] + predictors[i] * to_vector(a[slitlet_number[i]]);
    //     // tau[i] = sigma[slitlet_number[i]];
    // }

    wavelengths ~ normal(constant[slitlet_number] + rows_dot_product(a[slitlet_number,], predictors), sigma);
}

generated quantities {
    array[N] real wavelengths_ppc;
    for (i in 1:N){
        wavelengths_ppc[i] = normal_rng(constant[slitlet_number[i]] + predictors[i] * to_vector(a[slitlet_number[i]]), sigma);
        wavelengths_ppc[i] = wavelengths_ppc[i] * wavelengths_std + wavelengths_mean;
    }
}