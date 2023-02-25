/*
 * initial_value.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_INITIAL_VALUE_HPP_
#define INCLUDE_INITIAL_VALUE_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;

template <int dim> class InitialValue : public Function<dim>
{
public:
  InitialValue() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;

};

template <int dim>
double InitialValue<dim>::value(const Point<dim> &p,
                                const unsigned int /*component*/) const
{
	double return_value = 0;


//  for (unsigned int d = 0; d < dim; ++d)
//	    return_value += (p(d) - 0.5) * (p(d) - 0.5);


  return return_value;


}

template <int dim>
void InitialValue<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<double> &values,
                                   const unsigned int /*component = 0*/) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
  {
	  values[p] = value(points[p]);
  } // end ++p
}

} // end namespace Coefficients

#endif /* INCLUDE_INITIAL_VALUE_HPP_ */
