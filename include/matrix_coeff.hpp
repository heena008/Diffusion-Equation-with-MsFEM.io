/*
 * matrix_coeff.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_MATRIX_COEFF_HPP_
#define INCLUDE_MATRIX_COEFF_HPP_

// Deal.ii
#include <deal.II/base/tensor_function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

/*!
 * @class MatrixCoeff
 * @brief Diffusion coefficient.
 *
 * Class implements a matrix valued diffusion coefficient.
 * This coefficient must be positive definite.
 */

template <int dim> class MatrixCoeff : public TensorFunction<2, dim>
{
public:
  MatrixCoeff() : TensorFunction<2, dim>() {}

  virtual Tensor<2, dim> value(const Point<dim> &point) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<Tensor<2, dim>> &values) const override;


  const bool is_transient = false;

private:
  const int    k            = 240;
  const double scale_factor = 0.9999;


};

template <int dim>
Tensor<2, dim> MatrixCoeff<dim>::value(const Point<dim> &p) const
{
  Tensor<2, dim> value;
  value.clear();

  const double t = this->get_time();

  for (unsigned int d = 0; d < dim; ++d)
  {
	  using numbers::PI;
//	  value[d][d] =  1.0 * (1.0 - scale_factor *
//              (0.5 * sin(2 * PI_D * k * p(0)) +
//               0.5 * sin(2 * PI_D * k *
//                         p(1))));  /* Must be positive definite. For non-periodic case*/
     value[d][d] =1*(1- sin(2 * PI* p(d)/0.05));


 /* Must be positive
    //  definite.For periodic case */
  }

//  Point<dim> new_position = p;
//
//  double dtheta = -45;
//
//
//     	         new_position[0] = std::cos(dtheta) * p[0] - std::sin(dtheta) * p[1];
//     	         new_position[1] = std::sin(dtheta) * p[0] + std::cos(dtheta) * p[1];

//  for (unsigned int d = 0; d < dim; ++d)
//   {
////
//}

  return value;
}

template <int dim>
void MatrixCoeff<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<2, dim>> &values) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
  {
    values[p].clear();
    for (unsigned int d = 0; d < dim; ++d)
      {
    	using numbers::PI;
    values[p][d][d] =1*(1- sin(2 * PI * 2* points[p](d)/0.05));
      }
  }
}

} // end namespace Coefficients
#endif /* INCLUDE_MATRIX_COEFF_HPP_ */
