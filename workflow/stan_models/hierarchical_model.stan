data{
    int<lower=1> N; // Number of data points
    int<lower=1> N_fibres; // Number of fibres
    int<lower=1> x_polynomial_order; // Number of fibres
    //int<lower=1> y_polynomial_order; // Number of fibres
    vector[N] wavelengths;
    array[N] int fibre_numbers;
    matrix[N, x_polynomial_order + 1] x_values;
    //matrix[N, x_polynomial_order + 1] x_errors;
    //matrix[N_fibres, y_polynomial_order + 1] y_values;
}

parameters{

    matrix[N_fibres, x_polynomial_order + 1] a;
    //matrix[N_fibres, y_polynomial_order + 1] b;
    real<lower=0> sigma;
    //real<lower=1> nu;
    real phi;
    real<lower=0> tau;
    real theta;
    real<lower=0> nu;
}

transformed parameters {
   
//    matrix[N_fibres, x_polynomial_order + 1] a;

//     for (i in 1:N_fibres){
//         a[i, 1] = phi + tau * a_raw[i, 1];
//         a[i, 2] = theta + nu * a_raw[i, 2];
//         a[i, 3] = phi + tau * a_raw[i, 3];
//         a[i, 4] = phi + tau * a_raw[i, 4];
//     }
}


model{

    vector[N] mu;
    // sigma_scaled;

   for (i in 1:N){
        mu[i] = dot_product(a[fibre_numbers[i]], x_values[i]);
        //sigma_scaled[i] = 0.001 * sigma[fibre_numbers[i]];
   }

//    phi ~ std_normal();
//    tau ~ normal(0, 10);
//    theta ~ normal(1, 1);
    //nu ~  gamma(2, 0.1);

   
   //mu ~ std_normal();
   
    for (i in 1:N_fibres){
        //sigma[i] ~ std_normal();
        // for (j in 1:x_polynomial_order){
        //     a[i, j] ~ std_normal();
        // }
        a[i, 1] ~ normal(phi, tau);
        a[i, 2] ~ normal(theta, nu);
        a[i, 3] ~ normal(phi, tau);
        a[i, 4] ~ normal(phi, tau);
    }

    wavelengths ~ normal(mu, sigma * 0.001);
    

}