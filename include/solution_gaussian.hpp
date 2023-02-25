/*
 * solution_gaussian.hpp
 *
 *  Created on: Jun 24, 2020
 *      Author: heena
 */

#ifndef DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_
#define DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_


// Deal.ii
#include <deal.II/base/function.h>
// My Headers
#include "coefficients.h"
// STL
#include <cmath>
#include <fstream>


using namespace dealii;

  template <int dim>
  class SolutionBase
  {
  protected:
    static const std::array<Point<dim>, 3> source_centers;
    static const double                    width;

  };


  template <>
  const std::array<Point<2>, 3> SolutionBase<2>::source_centers = {
    {Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)}};

  template <>
  const std::array<Point<3>, 3> SolutionBase<3>::source_centers = {
    {Point<3>(-0.5, +0.0,0.5), Point<3>(-0.5, -0.5,0), Point<3>(0.0, -0.5,0.5)}};

  template <int dim>
  const double SolutionBase<dim>::width = 1. / 8.;



  template <int dim>
  class Solution : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;


  };


  template <int dim>
  double Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    double return_value = 0;
//    for (const auto &center : this->source_centers)
//      {
//        const Tensor<1, dim> x_minus_xi = p - center;
//        return_value +=
//          std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
//      }
    	using numbers::PI;

        return_value =
        		 std::sin(2. * PI * p(0)) * std::sin(2. * PI * p(1));


    return return_value;
  }


  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                         const unsigned int) const
  {
    Tensor<1, dim> return_value;
//    for (const auto &center : this->source_centers)
//      {
//        const Tensor<1, dim> x_minus_xi = p - center;
//
//        return_value +=
//          (-2. / (this->width * this->width) *
//           std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) *
//           x_minus_xi);
//      }
    using numbers::PI;
    return_value[0] =
      2. * PI * std::cos(2. * PI * p[0]) * std::sin(2. * PI * p[1]);
    return_value[1] =
      2. * PI * std::sin(2. * PI * p[0]) * std::cos(2. * PI * p[1]);


    return return_value;
  }





#endif /* DIFFUSION_PROBLEM_INCLUDE_SOLUTION_GAUSSIAN_HPP_ */
