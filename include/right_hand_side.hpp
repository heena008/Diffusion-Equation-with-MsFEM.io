/*
 * right_hand_side.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_RIGHT_HAND_SIDE_HPP_
#define INCLUDE_RIGHT_HAND_SIDE_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients
{
using namespace dealii;

/*!
 * @class RightHandSide
 * @brief Class implements scalar right-hand side function.
 *
 * The right-hand side represents some external forcing parameter.
 */

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
   {Point<3>(-0.5, +0.5,0.5), Point<3>(-0.5, -0.5,-0.5), Point<3>(+0.5, -0.5,0.5)}};

 template <int dim>
 const double SolutionBase<dim>::width = 1. / 8.;

template <int dim>
class RightHandSide : public Function<dim>, protected SolutionBase<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;

  virtual void
     value_list(const std::vector<Point<dim>> &points,
                std::vector<double> &          values,
                const unsigned int             component = 0) const override;

};


template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int) const
{
  double return_value = 0;


	using numbers::PI;


 return std::sin(2. * PI * p(0)) *
                          std::sin(2. * PI * p(1));
      return return_value;
}

template <int dim>
 void
 RightHandSide<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &          values,
                                const unsigned int /*component = 0*/) const
 {
   Assert(points.size() == values.size(),
          ExcDimensionMismatch(points.size(), values.size()));

   using numbers::PI;
   for (unsigned int i = 0; i < values.size(); ++i)
     values[i] = 8. * PI * PI * std::sin(2. * PI * points[i][0]) *
                 std::sin(2. * PI * points[i][1]);

 }

} // end namespace Coefficients

#endif /* INCLUDE_RIGHT_HAND_SIDE_HPP_ */
