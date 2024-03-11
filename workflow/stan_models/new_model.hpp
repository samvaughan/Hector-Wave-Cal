// Code generated by stanc v2.33.1
#include <stan/model/model_header.hpp>
namespace new_model_model_namespace {
using stan::model::model_base_crtp;
using namespace stan::math;
stan::math::profile_map profiles__;
static constexpr std::array<const char*, 31> locations_array__ =
  {" (found before start of program)",
  " (in 'new_model.stan', line 15, column 4 to column 37)",
  " (in 'new_model.stan', line 16, column 4 to column 31)",
  " (in 'new_model.stan', line 17, column 4 to column 24)",
  " (in 'new_model.stan', line 32, column 4 to column 34)",
  " (in 'new_model.stan', line 34, column 8 to column 126)",
  " (in 'new_model.stan', line 35, column 8 to column 85)",
  " (in 'new_model.stan', line 33, column 18 to line 36, column 5)",
  " (in 'new_model.stan', line 33, column 4 to line 36, column 5)",
  " (in 'new_model.stan', line 23, column 11 to column 12)",
  " (in 'new_model.stan', line 23, column 4 to column 17)",
  " (in 'new_model.stan', line 25, column 8 to column 94)",
  " (in 'new_model.stan', line 24, column 18 to line 26, column 5)",
  " (in 'new_model.stan', line 24, column 4 to line 26, column 5)",
  " (in 'new_model.stan', line 27, column 4 to column 36)",
  " (in 'new_model.stan', line 2, column 4 to column 19)",
  " (in 'new_model.stan', line 3, column 4 to column 26)",
  " (in 'new_model.stan', line 4, column 4 to column 30)",
  " (in 'new_model.stan', line 5, column 11 to column 12)",
  " (in 'new_model.stan', line 5, column 4 to column 26)",
  " (in 'new_model.stan', line 6, column 10 to column 11)",
  " (in 'new_model.stan', line 6, column 4 to column 31)",
  " (in 'new_model.stan', line 7, column 11 to column 12)",
  " (in 'new_model.stan', line 7, column 14 to column 26)",
  " (in 'new_model.stan', line 7, column 4 to column 39)",
  " (in 'new_model.stan', line 8, column 4 to column 25)",
  " (in 'new_model.stan', line 9, column 4 to column 26)",
  " (in 'new_model.stan', line 15, column 11 to column 19)",
  " (in 'new_model.stan', line 15, column 21 to column 33)",
  " (in 'new_model.stan', line 16, column 11 to column 19)",
  " (in 'new_model.stan', line 32, column 10 to column 11)"};
class new_model_model final : public model_base_crtp<new_model_model> {
 private:
  int N;
  int N_fibres;
  int N_predictors;
  Eigen::Matrix<double,-1,1> wavelengths_data__;
  std::vector<int> fibre_numbers;
  Eigen::Matrix<double,-1,-1> predictors_data__;
  double wavelengths_std;
  double wavelengths_mean;
  Eigen::Map<Eigen::Matrix<double,-1,1>> wavelengths{nullptr, 0};
  Eigen::Map<Eigen::Matrix<double,-1,-1>> predictors{nullptr, 0, 0};
 public:
  ~new_model_model() {}
  new_model_model(stan::io::var_context& context__, unsigned int
                  random_seed__ = 0, std::ostream* pstream__ = nullptr)
      : model_base_crtp(0) {
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    using local_scalar_t__ = double;
    boost::ecuyer1988 base_rng__ =
      stan::services::util::create_rng(random_seed__, 0);
    // suppress unused var warning
    (void) base_rng__;
    static constexpr const char* function__ =
      "new_model_model_namespace::new_model_model";
    // suppress unused var warning
    (void) function__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 15;
      context__.validate_dims("data initialization", "N", "int",
        std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      current_statement__ = 15;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 15;
      stan::math::check_greater_or_equal(function__, "N", N, 1);
      current_statement__ = 16;
      context__.validate_dims("data initialization", "N_fibres", "int",
        std::vector<size_t>{});
      N_fibres = std::numeric_limits<int>::min();
      current_statement__ = 16;
      N_fibres = context__.vals_i("N_fibres")[(1 - 1)];
      current_statement__ = 16;
      stan::math::check_greater_or_equal(function__, "N_fibres", N_fibres, 1);
      current_statement__ = 17;
      context__.validate_dims("data initialization", "N_predictors", "int",
        std::vector<size_t>{});
      N_predictors = std::numeric_limits<int>::min();
      current_statement__ = 17;
      N_predictors = context__.vals_i("N_predictors")[(1 - 1)];
      current_statement__ = 17;
      stan::math::check_greater_or_equal(function__, "N_predictors",
        N_predictors, 1);
      current_statement__ = 18;
      stan::math::validate_non_negative_index("wavelengths", "N", N);
      current_statement__ = 19;
      context__.validate_dims("data initialization", "wavelengths", "double",
        std::vector<size_t>{static_cast<size_t>(N)});
      wavelengths_data__ = Eigen::Matrix<double,-1,1>::Constant(N,
                             std::numeric_limits<double>::quiet_NaN());
      new (&wavelengths)
        Eigen::Map<Eigen::Matrix<double,-1,1>>(wavelengths_data__.data(), N);
      {
        std::vector<local_scalar_t__> wavelengths_flat__;
        current_statement__ = 19;
        wavelengths_flat__ = context__.vals_r("wavelengths");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
          stan::model::assign(wavelengths, wavelengths_flat__[(pos__ - 1)],
            "assigning variable wavelengths", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      current_statement__ = 20;
      stan::math::validate_non_negative_index("fibre_numbers", "N", N);
      current_statement__ = 21;
      context__.validate_dims("data initialization", "fibre_numbers", "int",
        std::vector<size_t>{static_cast<size_t>(N)});
      fibre_numbers = std::vector<int>(N, std::numeric_limits<int>::min());
      current_statement__ = 21;
      fibre_numbers = context__.vals_i("fibre_numbers");
      current_statement__ = 22;
      stan::math::validate_non_negative_index("predictors", "N", N);
      current_statement__ = 23;
      stan::math::validate_non_negative_index("predictors", "N_predictors",
        N_predictors);
      current_statement__ = 24;
      context__.validate_dims("data initialization", "predictors", "double",
        std::vector<size_t>{static_cast<size_t>(N),
          static_cast<size_t>(N_predictors)});
      predictors_data__ = Eigen::Matrix<double,-1,-1>::Constant(N,
                            N_predictors,
                            std::numeric_limits<double>::quiet_NaN());
      new (&predictors)
        Eigen::Map<Eigen::Matrix<double,-1,-1>>(predictors_data__.data(), N,
        N_predictors);
      {
        std::vector<local_scalar_t__> predictors_flat__;
        current_statement__ = 24;
        predictors_flat__ = context__.vals_r("predictors");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= N_predictors; ++sym1__) {
          for (int sym2__ = 1; sym2__ <= N; ++sym2__) {
            stan::model::assign(predictors, predictors_flat__[(pos__ - 1)],
              "assigning variable predictors",
              stan::model::index_uni(sym2__), stan::model::index_uni(sym1__));
            pos__ = (pos__ + 1);
          }
        }
      }
      current_statement__ = 25;
      context__.validate_dims("data initialization", "wavelengths_std",
        "double", std::vector<size_t>{});
      wavelengths_std = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 25;
      wavelengths_std = context__.vals_r("wavelengths_std")[(1 - 1)];
      current_statement__ = 26;
      context__.validate_dims("data initialization", "wavelengths_mean",
        "double", std::vector<size_t>{});
      wavelengths_mean = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 26;
      wavelengths_mean = context__.vals_r("wavelengths_mean")[(1 - 1)];
      current_statement__ = 27;
      stan::math::validate_non_negative_index("a", "N_fibres", N_fibres);
      current_statement__ = 28;
      stan::math::validate_non_negative_index("a", "N_predictors",
        N_predictors);
      current_statement__ = 29;
      stan::math::validate_non_negative_index("constants", "N_fibres",
        N_fibres);
      current_statement__ = 30;
      stan::math::validate_non_negative_index("wavelengths_ppc", "N", N);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = (N_fibres * N_predictors) + N_fibres + 1;
  }
  inline std::string model_name() const final {
    return "new_model_model";
  }
  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.33.1",
             "stancflags = --filename-in-msg=new_model.stan"};
  }
  // Base log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_not_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "new_model_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,-1> a =
        Eigen::Matrix<local_scalar_t__,-1,-1>::Constant(N_fibres,
          N_predictors, DUMMY_VAR__);
      current_statement__ = 1;
      a = in__.template read<Eigen::Matrix<local_scalar_t__,-1,-1>>(N_fibres,
            N_predictors);
      Eigen::Matrix<local_scalar_t__,-1,1> constants =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N_fibres, DUMMY_VAR__);
      current_statement__ = 2;
      constants = in__.template read<
                    Eigen::Matrix<local_scalar_t__,-1,1>>(N_fibres);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 3;
      sigma = in__.template read_constrain_lb<local_scalar_t__,
                jacobian__>(0, lp__);
      {
        current_statement__ = 9;
        stan::math::validate_non_negative_index("mu", "N", N);
        Eigen::Matrix<local_scalar_t__,-1,1> mu =
          Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N, DUMMY_VAR__);
        current_statement__ = 13;
        for (int i = 1; i <= N; ++i) {
          current_statement__ = 11;
          stan::model::assign(mu,
            (stan::model::rvalue(constants, "constants",
               stan::model::index_uni(
                 stan::model::rvalue(fibre_numbers, "fibre_numbers",
                   stan::model::index_uni(i)))) +
            stan::math::dot_product(
              stan::model::rvalue(a, "a",
                stan::model::index_uni(
                  stan::model::rvalue(fibre_numbers, "fibre_numbers",
                    stan::model::index_uni(i)))),
              stan::model::rvalue(predictors, "predictors",
                stan::model::index_uni(i)))), "assigning variable mu",
            stan::model::index_uni(i));
        }
        current_statement__ = 14;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(wavelengths, mu,
                         sigma));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  // Reverse mode autodiff log prob
  template <bool propto__, bool jacobian__, typename VecR, typename VecI,
            stan::require_vector_like_t<VecR>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr,
            stan::require_st_var<VecR>* = nullptr>
  inline stan::scalar_type_t<VecR>
  log_prob_impl(VecR& params_r__, VecI& params_i__, std::ostream*
                pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    static constexpr const char* function__ =
      "new_model_model_namespace::log_prob";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,-1> a =
        Eigen::Matrix<local_scalar_t__,-1,-1>::Constant(N_fibres,
          N_predictors, DUMMY_VAR__);
      current_statement__ = 1;
      a = in__.template read<Eigen::Matrix<local_scalar_t__,-1,-1>>(N_fibres,
            N_predictors);
      Eigen::Matrix<local_scalar_t__,-1,1> constants =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N_fibres, DUMMY_VAR__);
      current_statement__ = 2;
      constants = in__.template read<
                    Eigen::Matrix<local_scalar_t__,-1,1>>(N_fibres);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 3;
      sigma = in__.template read_constrain_lb<local_scalar_t__,
                jacobian__>(0, lp__);
      {
        current_statement__ = 9;
        stan::math::validate_non_negative_index("mu", "N", N);
        Eigen::Matrix<local_scalar_t__,-1,1> mu =
          Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N, DUMMY_VAR__);
        current_statement__ = 13;
        for (int i = 1; i <= N; ++i) {
          current_statement__ = 11;
          stan::model::assign(mu,
            (stan::model::rvalue(constants, "constants",
               stan::model::index_uni(
                 stan::model::rvalue(fibre_numbers, "fibre_numbers",
                   stan::model::index_uni(i)))) +
            stan::math::dot_product(
              stan::model::rvalue(a, "a",
                stan::model::index_uni(
                  stan::model::rvalue(fibre_numbers, "fibre_numbers",
                    stan::model::index_uni(i)))),
              stan::model::rvalue(predictors, "predictors",
                stan::model::index_uni(i)))), "assigning variable mu",
            stan::model::index_uni(i));
        }
        current_statement__ = 14;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(wavelengths, mu,
                         sigma));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
  }
  template <typename RNG, typename VecR, typename VecI, typename VecVar,
            stan::require_vector_like_vt<std::is_floating_point,
            VecR>* = nullptr, stan::require_vector_like_vt<std::is_integral,
            VecI>* = nullptr, stan::require_vector_vt<std::is_floating_point,
            VecVar>* = nullptr>
  inline void
  write_array_impl(RNG& base_rng__, VecR& params_r__, VecI& params_i__,
                   VecVar& vars__, const bool
                   emit_transformed_parameters__ = true, const bool
                   emit_generated_quantities__ = true, std::ostream*
                   pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    // suppress unused var warning
    (void) propto__;
    double lp__ = 0.0;
    // suppress unused var warning
    (void) lp__;
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    constexpr bool jacobian__ = false;
    // suppress unused var warning
    (void) jacobian__;
    static constexpr const char* function__ =
      "new_model_model_namespace::write_array";
    // suppress unused var warning
    (void) function__;
    try {
      Eigen::Matrix<double,-1,-1> a =
        Eigen::Matrix<double,-1,-1>::Constant(N_fibres, N_predictors,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      a = in__.template read<Eigen::Matrix<local_scalar_t__,-1,-1>>(N_fibres,
            N_predictors);
      Eigen::Matrix<double,-1,1> constants =
        Eigen::Matrix<double,-1,1>::Constant(N_fibres,
          std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 2;
      constants = in__.template read<
                    Eigen::Matrix<local_scalar_t__,-1,1>>(N_fibres);
      double sigma = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 3;
      sigma = in__.template read_constrain_lb<local_scalar_t__,
                jacobian__>(0, lp__);
      out__.write(a);
      out__.write(constants);
      out__.write(sigma);
      if (stan::math::logical_negation(
            (stan::math::primitive_value(emit_transformed_parameters__) ||
            stan::math::primitive_value(emit_generated_quantities__)))) {
        return ;
      }
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      }
      std::vector<double> wavelengths_ppc =
        std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 8;
      for (int i = 1; i <= N; ++i) {
        current_statement__ = 5;
        stan::model::assign(wavelengths_ppc,
          stan::math::normal_rng(
            (stan::model::rvalue(constants, "constants",
               stan::model::index_uni(
                 stan::model::rvalue(fibre_numbers, "fibre_numbers",
                   stan::model::index_uni(i)))) +
            stan::math::dot_product(
              stan::model::rvalue(a, "a",
                stan::model::index_uni(
                  stan::model::rvalue(fibre_numbers, "fibre_numbers",
                    stan::model::index_uni(i)))),
              stan::model::rvalue(predictors, "predictors",
                stan::model::index_uni(i)))), sigma, base_rng__),
          "assigning variable wavelengths_ppc", stan::model::index_uni(i));
        current_statement__ = 6;
        stan::model::assign(wavelengths_ppc,
          ((stan::model::rvalue(wavelengths_ppc, "wavelengths_ppc",
              stan::model::index_uni(i)) * wavelengths_std) +
          wavelengths_mean), "assigning variable wavelengths_ppc",
          stan::model::index_uni(i));
      }
      out__.write(wavelengths_ppc);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, typename VecI,
            stan::require_vector_t<VecVar>* = nullptr,
            stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr>
  inline void
  unconstrain_array_impl(const VecVar& params_r__, const VecI& params_i__,
                         VecVar& vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      Eigen::Matrix<local_scalar_t__,-1,-1> a =
        Eigen::Matrix<local_scalar_t__,-1,-1>::Constant(N_fibres,
          N_predictors, DUMMY_VAR__);
      current_statement__ = 1;
      stan::model::assign(a,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,-1>>(N_fibres,
          N_predictors), "assigning variable a");
      out__.write(a);
      Eigen::Matrix<local_scalar_t__,-1,1> constants =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N_fibres, DUMMY_VAR__);
      current_statement__ = 2;
      stan::model::assign(constants,
        in__.read<Eigen::Matrix<local_scalar_t__,-1,1>>(N_fibres),
        "assigning variable constants");
      out__.write(constants);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 3;
      sigma = in__.read<local_scalar_t__>();
      out__.write_free_lb(0, sigma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  template <typename VecVar, stan::require_vector_t<VecVar>* = nullptr>
  inline void
  transform_inits_impl(const stan::io::var_context& context__, VecVar&
                       vars__, std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    // suppress unused var warning
    (void) current_statement__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    // suppress unused var warning
    (void) DUMMY_VAR__;
    try {
      current_statement__ = 1;
      context__.validate_dims("parameter initialization", "a", "double",
        std::vector<size_t>{static_cast<size_t>(N_fibres),
          static_cast<size_t>(N_predictors)});
      current_statement__ = 2;
      context__.validate_dims("parameter initialization", "constants",
        "double", std::vector<size_t>{static_cast<size_t>(N_fibres)});
      current_statement__ = 3;
      context__.validate_dims("parameter initialization", "sigma", "double",
        std::vector<size_t>{});
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      Eigen::Matrix<local_scalar_t__,-1,-1> a =
        Eigen::Matrix<local_scalar_t__,-1,-1>::Constant(N_fibres,
          N_predictors, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> a_flat__;
        current_statement__ = 1;
        a_flat__ = context__.vals_r("a");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= N_predictors; ++sym1__) {
          for (int sym2__ = 1; sym2__ <= N_fibres; ++sym2__) {
            stan::model::assign(a, a_flat__[(pos__ - 1)],
              "assigning variable a", stan::model::index_uni(sym2__),
              stan::model::index_uni(sym1__));
            pos__ = (pos__ + 1);
          }
        }
      }
      out__.write(a);
      Eigen::Matrix<local_scalar_t__,-1,1> constants =
        Eigen::Matrix<local_scalar_t__,-1,1>::Constant(N_fibres, DUMMY_VAR__);
      {
        std::vector<local_scalar_t__> constants_flat__;
        current_statement__ = 2;
        constants_flat__ = context__.vals_r("constants");
        pos__ = 1;
        for (int sym1__ = 1; sym1__ <= N_fibres; ++sym1__) {
          stan::model::assign(constants, constants_flat__[(pos__ - 1)],
            "assigning variable constants", stan::model::index_uni(sym1__));
          pos__ = (pos__ + 1);
        }
      }
      out__.write(constants);
      local_scalar_t__ sigma = DUMMY_VAR__;
      current_statement__ = 3;
      sigma = context__.vals_r("sigma")[(1 - 1)];
      out__.write_free_lb(0, sigma);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
  }
  inline void
  get_param_names(std::vector<std::string>& names__, const bool
                  emit_transformed_parameters__ = true, const bool
                  emit_generated_quantities__ = true) const {
    names__ = std::vector<std::string>{"a", "constants", "sigma"};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::string> temp{"wavelengths_ppc"};
      names__.reserve(names__.size() + temp.size());
      names__.insert(names__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  get_dims(std::vector<std::vector<size_t>>& dimss__, const bool
           emit_transformed_parameters__ = true, const bool
           emit_generated_quantities__ = true) const {
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{static_cast<
                                                                    size_t>(
                                                                    N_fibres),
                                                 static_cast<size_t>(
                                                   N_predictors)},
                std::vector<size_t>{static_cast<size_t>(N_fibres)},
                std::vector<size_t>{}};
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      std::vector<std::vector<size_t>>
        temp{std::vector<size_t>{static_cast<size_t>(N)}};
      dimss__.reserve(dimss__.size() + temp.size());
      dimss__.insert(dimss__.end(), temp.begin(), temp.end());
    }
  }
  inline void
  constrained_param_names(std::vector<std::string>& param_names__, bool
                          emit_transformed_parameters__ = true, bool
                          emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= N_predictors; ++sym1__) {
      for (int sym2__ = 1; sym2__ <= N_fibres; ++sym2__) {
        param_names__.emplace_back(std::string() + "a" + '.' +
          std::to_string(sym2__) + '.' + std::to_string(sym1__));
      }
    }
    for (int sym1__ = 1; sym1__ <= N_fibres; ++sym1__) {
      param_names__.emplace_back(std::string() + "constants" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        param_names__.emplace_back(std::string() + "wavelengths_ppc" + '.' +
          std::to_string(sym1__));
      }
    }
  }
  inline void
  unconstrained_param_names(std::vector<std::string>& param_names__, bool
                            emit_transformed_parameters__ = true, bool
                            emit_generated_quantities__ = true) const final {
    for (int sym1__ = 1; sym1__ <= N_predictors; ++sym1__) {
      for (int sym2__ = 1; sym2__ <= N_fibres; ++sym2__) {
        param_names__.emplace_back(std::string() + "a" + '.' +
          std::to_string(sym2__) + '.' + std::to_string(sym1__));
      }
    }
    for (int sym1__ = 1; sym1__ <= N_fibres; ++sym1__) {
      param_names__.emplace_back(std::string() + "constants" + '.' +
        std::to_string(sym1__));
    }
    param_names__.emplace_back(std::string() + "sigma");
    if (emit_transformed_parameters__) {}
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        param_names__.emplace_back(std::string() + "wavelengths_ppc" + '.' +
          std::to_string(sym1__));
      }
    }
  }
  inline std::string get_constrained_sizedtypes() const {
    return std::string("[{\"name\":\"a\",\"type\":{\"name\":\"matrix\",\"rows\":" + std::to_string(N_fibres) + ",\"cols\":" + std::to_string(N_predictors) + "},\"block\":\"parameters\"},{\"name\":\"constants\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_fibres) + "},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"wavelengths_ppc\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
  }
  inline std::string get_unconstrained_sizedtypes() const {
    return std::string("[{\"name\":\"a\",\"type\":{\"name\":\"matrix\",\"rows\":" + std::to_string(N_fibres) + ",\"cols\":" + std::to_string(N_predictors) + "},\"block\":\"parameters\"},{\"name\":\"constants\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N_fibres) + "},\"block\":\"parameters\"},{\"name\":\"sigma\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"wavelengths_ppc\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(N) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
  }
  // Begin method overload boilerplate
  template <typename RNG> inline void
  write_array(RNG& base_rng, Eigen::Matrix<double,-1,1>& params_r,
              Eigen::Matrix<double,-1,1>& vars, const bool
              emit_transformed_parameters = true, const bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((N_fibres * N_predictors) + N_fibres) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (N);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    std::vector<int> params_i;
    vars = Eigen::Matrix<double,-1,1>::Constant(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <typename RNG> inline void
  write_array(RNG& base_rng, std::vector<double>& params_r, std::vector<int>&
              params_i, std::vector<double>& vars, bool
              emit_transformed_parameters = true, bool
              emit_generated_quantities = true, std::ostream*
              pstream = nullptr) const {
    const size_t num_params__ = (((N_fibres * N_predictors) + N_fibres) + 1);
    const size_t num_transformed = emit_transformed_parameters * (0);
    const size_t num_gen_quantities = emit_generated_quantities * (N);
    const size_t num_to_write = num_params__ + num_transformed +
      num_gen_quantities;
    vars = std::vector<double>(num_to_write,
             std::numeric_limits<double>::quiet_NaN());
    write_array_impl(base_rng, params_r, params_i, vars,
      emit_transformed_parameters, emit_generated_quantities, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(Eigen::Matrix<T_,-1,1>& params_r, std::ostream* pstream = nullptr) const {
    Eigen::Matrix<int,-1,1> params_i;
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  template <bool propto__, bool jacobian__, typename T_> inline T_
  log_prob(std::vector<T_>& params_r, std::vector<int>& params_i,
           std::ostream* pstream = nullptr) const {
    return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
  }
  inline void
  transform_inits(const stan::io::var_context& context,
                  Eigen::Matrix<double,-1,1>& params_r, std::ostream*
                  pstream = nullptr) const final {
    std::vector<double> params_r_vec(params_r.size());
    std::vector<int> params_i;
    transform_inits(context, params_i, params_r_vec, pstream);
    params_r = Eigen::Map<Eigen::Matrix<double,-1,1>>(params_r_vec.data(),
                 params_r_vec.size());
  }
  inline void
  transform_inits(const stan::io::var_context& context, std::vector<int>&
                  params_i, std::vector<double>& vars, std::ostream*
                  pstream__ = nullptr) const {
    vars.resize(num_params_r__);
    transform_inits_impl(context, vars, pstream__);
  }
  inline void
  unconstrain_array(const std::vector<double>& params_constrained,
                    std::vector<double>& params_unconstrained, std::ostream*
                    pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = std::vector<double>(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
  inline void
  unconstrain_array(const Eigen::Matrix<double,-1,1>& params_constrained,
                    Eigen::Matrix<double,-1,1>& params_unconstrained,
                    std::ostream* pstream = nullptr) const {
    const std::vector<int> params_i;
    params_unconstrained = Eigen::Matrix<double,-1,1>::Constant(num_params_r__,
                             std::numeric_limits<double>::quiet_NaN());
    unconstrain_array_impl(params_constrained, params_i,
      params_unconstrained, pstream);
  }
};
}
using stan_model = new_model_model_namespace::new_model_model;
#ifndef USING_R
// Boilerplate
stan::model::model_base&
new_model(stan::io::var_context& data_context, unsigned int seed,
          std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}
stan::math::profile_map& get_stan_profile_data() {
  return new_model_model_namespace::profiles__;
}
#endif