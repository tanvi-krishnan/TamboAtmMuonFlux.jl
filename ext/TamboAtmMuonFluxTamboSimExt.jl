module TamboAtmMuonFluxTamboSimExt

using TamboAtmMuonFlux
using TamboSim
using Unitful
import LinearAlgebra: norm, dot

"""
    Φ_μ(q::TamboSim.Frame) -> Float64

Atmospheric muon flux for a TamboSim Frame event (GeV⁻¹ cm⁻² s⁻¹ sr⁻¹).

Reads energy and direction from `q["injection_initial_state"]` and altitude
from `q["injection"]["altitude"]` (km), which is always snapshotted onto the
M frame by MuonInjection / CosmicRayInjection.
"""
function TamboAtmMuonFlux.Φ_μ(q::TamboSim.Frame)
    p     = q["injection_initial_state"]
    E_GeV = ustrip(u"GeV", p.energy)
    cosθ  = _local_zenith(p)
    h_km  = Float64(q["injection"]["altitude"])
    Φ_μ(E_GeV, cosθ, h_km)
end

# cos of local zenith: dot(-direction, r̂) where r̂ points radially outward
function _local_zenith(p)
    pos = p.position.point
    dir = p.direction.point
    r̂ = pos / norm(pos)
    clamp(-dot(dir, r̂), 0.0, 1.0)
end

end
