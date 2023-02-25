/*
 * dirichlet_bc.hpp
 *
 *  Created on: Oct 7, 2019
 *      Author: heena
 */

#ifndef INCLUDE_DIRICHLET_BC_HPP_
#define INCLUDE_DIRICHLET_BC_HPP_

// Deal.ii
#include <deal.II/base/function.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include "coefficients.h"

namespace Coefficients {
using namespace dealii;
/*!
 * @class DirichletBC
 * @brief Class implements Dirichlet conditions.
 */
template <int dim> class DirichletBC : public Function<dim>
{
public:
  DirichletBC() : Function<dim>() {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<double> &values,
                          const unsigned int component = 0) const override;

};

template <int dim>
double DirichletBC<dim>::value(const Point<dim> &p,
                               const unsigned int /*component*/) const
{
  double return_value = 0;

  for (unsigned int d = 0; d < dim; ++d)
  	    return_value = 0;//(p(d) - 0.5) * (p(d) - 0.5);


  return return_value;

}

template <int dim>
void DirichletBC<dim>::value_list(const std::vector<Point<dim>> &points,
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

#endif /* INCLUDE_DIRICHLET_BC_HPP_ */
