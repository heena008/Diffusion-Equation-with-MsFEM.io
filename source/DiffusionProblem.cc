/*
 * Diffusion.cc
 *
 *  Created on: Oct 14, 2019
 *      Author: heena
 */

#include "diffusion_problem.hpp"
#include "diffusion_problem_multiscale.hpp"
#include "adpative_diffusion_problem.hpp"

using namespace dealii;


int main(int argc, char *argv[])

{
  try
    {
	  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		
      const unsigned int n_refine =3, n_refine_local =5;

     const int dim = 2;

    	 DiffusionProblem::DiffusionProblem<dim> diffusion_problem_coarse(
            n_refine);
          diffusion_problem_coarse.run();

          DiffusionProblem::DiffusionProblem<dim> diffusion_problem_fine(
            n_refine + n_refine_local);
          diffusion_problem_fine.run();

         DiffusionProblem::DiffusionProblemAdpative<dim> diffusion_problem_adaptive(
                     n_refine );
          diffusion_problem_adaptive.run();

   	  DiffusionProblem::DiffusionProblemMultiscale<dim>
            diffusion_ms_problem(n_refine, n_refine_local);
          diffusion_ms_problem.run();


    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
