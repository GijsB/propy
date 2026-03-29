from typing import Any, Callable, Iterable, TYPE_CHECKING, Protocol
from dataclasses import dataclass

from scipy.optimize import minimize, Bounds

if TYPE_CHECKING:
    from propy.propeller import Propeller


@dataclass(frozen=True)
class FunctionWrapper:
    base: "Propeller"
    func: Callable[["Propeller"], float]

    def __call__(self, args: Iterable[float]) -> float:
        args = tuple(float(arg) for arg in args)
        return self.func(self.base.new(self.base.blades, *args))


class OptimizationMethod(Protocol):
    def __call__(
        self,
        objective: FunctionWrapper,
        constraints: Iterable[FunctionWrapper],
        bounds: Iterable[tuple[float, float, float]],
        verbose: bool = False
    ) -> Any: ...


def slsqp(
    objective: FunctionWrapper,
    constraints: Iterable[FunctionWrapper],
    bounds: Iterable[tuple[float, float, float]],
    verbose: bool = False,
) -> tuple[float, ...]:
    
    opt_res = minimize(
        method='SLSQP',
        fun=objective,
        x0=tuple(bound[1] for bound in bounds),
        bounds=Bounds(
            lb=tuple(bound[0] for bound in bounds),
            ub=tuple(bound[2] for bound in bounds),
            keep_feasible=tuple([True for _ in bounds])
        ),
        constraints=[{'type': 'ineq', 'fun': cfun} for cfun in constraints]
    )

    if verbose:
        print(opt_res)

    if not opt_res.success:
        raise RuntimeError(opt_res.message)
    
    return tuple(float(arg) for arg in opt_res.x)


def trust_constrained(
    objective: FunctionWrapper,
    constraints: Iterable[FunctionWrapper],
    bounds: Iterable[tuple[float, float, float]],
    verbose: bool = False,
) -> tuple[float, ...]:

    opt_res = minimize(
        method='trust-constr',
        fun=objective,
        x0=tuple(bound[1] for bound in bounds),
        bounds=Bounds(
            lb=tuple(bound[0] for bound in bounds),
            ub=tuple(bound[2] for bound in bounds),
            keep_feasible=(True, True, True)
        ),
        constraints=[{'type': 'ineq', 'fun': cfun} for cfun in constraints]
    )

    if verbose:
        print(opt_res)

    if not opt_res.success:
        raise RuntimeError(opt_res.message)
    
    return tuple(float(arg) for arg in opt_res.x)
